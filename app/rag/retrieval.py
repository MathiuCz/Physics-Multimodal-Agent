"""
Capa de recuperación: encuentra las diapositivas candidatas para una consulta.

Modos para consultas de texto (seleccionables para poder compararlos en la evaluación):
- "clip":    texto CLIP contra el vector de imagen de la diapositiva (línea base original).
- "texto":   búsqueda semántica con bge-m3 sobre el documento de búsqueda (texto + descripción).
- "lexico":  BM25 sobre el mismo documento de búsqueda.
- "hibrido": "texto" + "lexico" fusionados con Reciprocal Rank Fusion (por defecto).

Las consultas con imagen siempre usan CLIP (imagen contra imagen). Si llegan texto e imagen,
se fusionan ambas listas con RRF.

La usan el grafo del tutor y el script de evaluación, para que ambos midan
exactamente el mismo recuperador.
"""

import base64
import logging
from typing import Any, Literal, Optional

from app.core.supabase_client import client
from app.rag.embedding import generar_embedding, generar_embedding_imagen, generar_embeddings_texto
from app.rag.lexico import indice_bm25

logger = logging.getLogger(__name__)

ModoRecuperacion = Literal["clip", "texto", "lexico", "hibrido"]
MODOS: tuple[str, ...] = ("clip", "texto", "lexico", "hibrido")
MODO_POR_DEFECTO: ModoRecuperacion = "hibrido"

# Se piden más filas de las necesarias para tener margen tras deduplicar
# y para que la fusión RRF tenga candidatas de ambas listas.
FACTOR_SOBREMUESTREO = 3
K_RRF = 60  # constante estándar de RRF: amortigua la ventaja de las primeras posiciones


def vectorizar_consulta(
    consulta: Optional[str] = None,
    image_base64: Optional[str] = None,
) -> list[float]:
    """Devuelve el vector CLIP de la consulta. La imagen tiene prioridad sobre el texto."""
    if image_base64:
        datos = image_base64.split(",")[-1]  # quita el prefijo "data:image/...;base64,"
        vector = generar_embedding_imagen(base64.b64decode(datos))
    elif consulta:
        vector = generar_embedding(consulta)
    else:
        raise ValueError("Debes enviar un texto o una imagen.")

    if not vector:
        raise ValueError("No se pudo generar el vector de la consulta.")
    return vector


def _clave(doc: dict[str, Any]) -> tuple[Any, Any]:
    meta = doc.get("metadata") or {}
    return (meta.get("archivo"), meta.get("diapositiva"))


def deduplicar(documentos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Conserva la primera aparición de cada (archivo, diapositiva), respetando el orden."""
    vistos: set[tuple[Any, Any]] = set()
    unicos: list[dict[str, Any]] = []
    for doc in documentos:
        clave = _clave(doc)
        if clave in vistos:
            continue
        vistos.add(clave)
        unicos.append(doc)
    return unicos


def _limpiar(documentos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Los vectores no le sirven a nadie aguas abajo y engordan las respuestas.
    for doc in documentos:
        doc.pop("embedding", None)
        doc.pop("embedding_texto", None)
    return documentos


# ---------- Recuperadores individuales ----------
def buscar_diapositivas(vector: list[float], top_k: int = 3) -> list[dict[str, Any]]:
    """CLIP: vector (de texto o imagen) contra el vector de imagen de cada diapositiva."""
    resultado = client.rpc(
        "buscar_similares",
        {"query": vector, "top_k": top_k * FACTOR_SOBREMUESTREO},
    ).execute()
    return _limpiar(deduplicar(resultado.data or []))[:top_k]


def buscar_semantico_texto(consulta: str, top_k: int = 3) -> list[dict[str, Any]]:
    """bge-m3: consulta contra el documento de búsqueda de cada diapositiva."""
    vector = generar_embeddings_texto([consulta])[0]
    resultado = client.rpc(
        "buscar_similares_texto",
        {"query": vector, "top_k": top_k * FACTOR_SOBREMUESTREO},
    ).execute()
    return _limpiar(deduplicar(resultado.data or []))[:top_k]


def buscar_lexico(consulta: str, top_k: int = 3) -> list[dict[str, Any]]:
    """BM25 en memoria sobre el documento de búsqueda."""
    return _limpiar(deduplicar(indice_bm25().buscar(consulta, top_k * FACTOR_SOBREMUESTREO)))[:top_k]


def fusionar_rrf(*rankings: list[dict[str, Any]], top_k: int, k: int = K_RRF) -> list[dict[str, Any]]:
    """
    Reciprocal Rank Fusion: puntaje = Σ 1 / (k + posición) sobre cada lista donde aparece.

    Usa solo posiciones, no puntajes: así se pueden combinar BM25 (sin tope) y coseno
    (entre -1 y 1) sin calibrar escalas.
    """
    puntajes: dict[tuple[Any, Any], float] = {}
    documentos: dict[tuple[Any, Any], dict[str, Any]] = {}
    for ranking in rankings:
        for posicion, doc in enumerate(ranking, start=1):
            clave = _clave(doc)
            puntajes[clave] = puntajes.get(clave, 0.0) + 1.0 / (k + posicion)
            documentos.setdefault(clave, doc)

    ordenadas = sorted(puntajes, key=puntajes.get, reverse=True)[:top_k]
    return [{**documentos[c], "puntaje_rrf": puntajes[c]} for c in ordenadas]


def buscar_texto(consulta: str, top_k: int, modo: ModoRecuperacion) -> list[dict[str, Any]]:
    candidatas = top_k * FACTOR_SOBREMUESTREO
    if modo == "clip":
        return buscar_diapositivas(generar_embedding(consulta), top_k)
    if modo == "lexico":
        return buscar_lexico(consulta, top_k)
    if modo not in ("texto", "hibrido"):
        raise ValueError(f"Modo de recuperación desconocido: {modo}")

    semanticos = buscar_semantico_texto(consulta, candidatas if modo == "hibrido" else top_k)
    if not semanticos:
        # Degradación segura: la RPC existe pero el índice de texto aún no se ha poblado
        # (ingesta v2 pendiente). Mejor una recuperación peor que ninguna.
        logger.warning("Índice de texto vacío: el modo '%s' degrada a 'clip'.", modo)
        return buscar_diapositivas(generar_embedding(consulta), top_k)
    if modo == "texto":
        return semanticos
    return fusionar_rrf(semanticos, buscar_lexico(consulta, candidatas), top_k=top_k)


# ---------- Punto de entrada ----------
def recuperar(
    consulta: Optional[str] = None,
    image_base64: Optional[str] = None,
    top_k: int = 3,
    modo: ModoRecuperacion = MODO_POR_DEFECTO,
) -> list[dict[str, Any]]:
    """Enruta por modalidad: imagen → CLIP; texto → `modo`; ambos → fusión RRF."""
    if not consulta and not image_base64:
        raise ValueError("Debes enviar un texto o una imagen.")

    if image_base64:
        por_imagen = buscar_diapositivas(vectorizar_consulta(image_base64=image_base64), top_k * FACTOR_SOBREMUESTREO)
        if not consulta:
            return por_imagen[:top_k]
        por_texto = buscar_texto(consulta, top_k * FACTOR_SOBREMUESTREO, modo)
        return fusionar_rrf(por_imagen, por_texto, top_k=top_k)

    return buscar_texto(consulta, top_k, modo)
