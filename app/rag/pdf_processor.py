"""
Ingesta de diapositivas: renderiza cada página de los PDFs de docs/ y la sincroniza con la
tabla `documentos` de Supabase con dos representaciones para BUSCAR:
- `embedding`: vector CLIP de la imagen (para consultas con imagen).
- `contenido` + `embedding_texto`: un documento de búsqueda en texto (contextual retrieval:
  título del PDF + contexto de la diapositiva anterior + descripción de Gemini + texto extraído),
  vectorizado con bge-m3. Sobre `contenido` también se construye el índice léxico BM25.
Para RESPONDER se sigue usando la imagen original del PDF (ver graph.py): la descripción
solo sirve para encontrar la diapositiva.

La ingesta es idempotente e incremental:
- ID determinista (uuid5 de archivo#diapositiva): re-ingestar actualiza, no duplica.
- Huella del PDF (sha256) en metadata.hash_archivo: si el PDF no cambió, se omite.
- Por cada PDF re-ingestado se borran sus filas sobrantes (páginas eliminadas,
  duplicados antiguos con uuid4); también se borran los PDFs que ya no están en docs/.
- Primero upsert, después borrado: nunca hay un momento en que el PDF desaparezca de la base.
- metadata.version_indice: si cambia la forma de construir el índice (prompt, modelo),
  se sube la versión y la próxima sincronización re-ingesta todo.
- Las descripciones respetan la cuota de Gemini con una pausa entre llamadas (--pausa-llm).

Uso (desde la raíz):
    uv run python -m app.rag.pdf_processor              # sincroniza una vez
    uv run python -m app.rag.pdf_processor --forzar     # re-ingesta todo aunque no haya cambios
    uv run python -m app.rag.pdf_processor --watch      # vigila docs/ y sincroniza al detectar cambios
"""

import argparse
import base64
import hashlib
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import fitz

from app.core.supabase_client import client as supabase
from app.rag.embedding import generar_embedding_imagen, generar_embeddings_texto
from app.rag.llm import mensaje_multimodal, obtener_llm_auxiliar, texto_de
from app.rag.prompts import SYSTEM_PROMPT_DESCRIPCION

logger = logging.getLogger(__name__)

TABLA = "documentos"
TIPO_DIAPOSITIVA = "imagen"
TAMANO_PAGINA_SUPABASE = 1000   # máximo de filas que devuelve una consulta por defecto
TAMANO_LOTE_BORRADO = 200

# Subir esta versión cuando cambie cómo se construye el índice (prompt, modelo de texto...).
VERSION_INDICE = "v2-texto-bge-m3"
# Pausa entre descripciones para no saturar la cuota por minuto del modelo auxiliar (Flash-Lite).
# Si aun así llega un 429, el cliente reintenta con backoff y la caché evita repetir trabajo.
PAUSA_LLM_SEGUNDOS = 6.0

# Caché de descripciones direccionada por contenido: clave = hash del PDF + página + versión del índice.
# Renombrar un PDF o reintentar tras un fallo reutiliza las descripciones ya pagadas.
RUTA_CACHE_DESCRIPCIONES = os.path.join("cache", "descripciones.json")

# Namespace fijo del proyecto: mismo (archivo, diapositiva) → mismo UUID en cualquier máquina.
NAMESPACE_DIAPOSITIVAS = uuid.uuid5(uuid.NAMESPACE_URL, "api_motor_semantico/diapositivas")


def id_diapositiva(nombre_archivo: str, numero: int) -> str:
    """ID determinista de una diapositiva: re-ingestar actualiza la fila en vez de duplicarla."""
    return str(uuid.uuid5(NAMESPACE_DIAPOSITIVAS, f"{nombre_archivo}#{numero}"))


def hash_archivo(pdf_bytes: bytes) -> str:
    """Huella del contenido del PDF: cambia si cambia cualquier byte."""
    return hashlib.sha256(pdf_bytes).hexdigest()


# ---------- Estado local y remoto ----------
@dataclass
class ArchivoLocal:
    nombre: str
    ruta: str
    hash: str
    paginas: int


@dataclass
class PlanSincronizacion:
    ingestar: list[str] = field(default_factory=list)   # nuevos o modificados
    omitir: list[str] = field(default_factory=list)     # sin cambios
    huerfanos: list[str] = field(default_factory=list)  # en la base pero no en docs/


def escanear_directorio(ruta_carpeta: str) -> dict[str, ArchivoLocal]:
    """Lee los PDFs de la carpeta y calcula su huella y número de páginas."""
    locales: dict[str, ArchivoLocal] = {}
    for nombre in sorted(os.listdir(ruta_carpeta)):
        if not nombre.lower().endswith(".pdf"):
            continue
        ruta = os.path.join(ruta_carpeta, nombre)
        with open(ruta, "rb") as f:
            contenido = f.read()
        with fitz.open(stream=contenido, filetype="pdf") as doc:
            paginas = len(doc)
        locales[nombre] = ArchivoLocal(nombre, ruta, hash_archivo(contenido), paginas)
    return locales


def indice_remoto() -> dict[str, list[dict[str, Any]]]:
    """Filas de diapositivas en Supabase agrupadas por archivo (solo id y metadata, sin vectores)."""
    filas: list[dict[str, Any]] = []
    inicio = 0
    while True:
        lote = (
            supabase.table(TABLA)
            .select("id, metadata")
            .eq("tipo", TIPO_DIAPOSITIVA)
            .range(inicio, inicio + TAMANO_PAGINA_SUPABASE - 1)
            .execute()
            .data
            or []
        )
        filas.extend(lote)
        if len(lote) < TAMANO_PAGINA_SUPABASE:
            break
        inicio += TAMANO_PAGINA_SUPABASE

    indice: dict[str, list[dict[str, Any]]] = {}
    for fila in filas:
        archivo = (fila.get("metadata") or {}).get("archivo")
        if archivo:
            indice.setdefault(archivo, []).append(fila)
    return indice


def planificar_sincronizacion(
    locales: dict[str, ArchivoLocal],
    remoto: dict[str, list[dict[str, Any]]],
    forzar: bool = False,
) -> PlanSincronizacion:
    """
    Decide qué hacer con cada archivo sin tocar la base (función pura, fácil de testear).

    Un archivo se omite solo si en la base tiene exactamente una fila por página, todas
    con el hash y la versión de índice actuales. Filas sin hash o de otra versión (ingestas
    antiguas) o duplicadas fuerzan re-ingesta.
    """
    plan = PlanSincronizacion()
    for nombre, local in locales.items():
        filas = remoto.get(nombre, [])
        hashes = {(f.get("metadata") or {}).get("hash_archivo") for f in filas}
        versiones = {(f.get("metadata") or {}).get("version_indice") for f in filas}
        ids_esperados = {id_diapositiva(nombre, n) for n in range(1, local.paginas + 1)}
        al_dia = (
            hashes == {local.hash}
            and versiones == {VERSION_INDICE}
            and {f["id"] for f in filas} == ids_esperados
        )

        if al_dia and not forzar:
            plan.omitir.append(nombre)
        else:
            plan.ingestar.append(nombre)

    plan.huerfanos = sorted(set(remoto) - set(locales))
    return plan


# ---------- Caché de descripciones ----------
def clave_descripcion(hash_pdf: str, numero: int) -> str:
    return f"{hash_pdf}#{numero}#{VERSION_INDICE}"


def cargar_cache_descripciones() -> dict[str, str]:
    try:
        with open(RUTA_CACHE_DESCRIPCIONES, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def guardar_cache_descripciones(cache: dict[str, str]) -> None:
    """Escritura atómica: se escribe a un temporal y se reemplaza, para no dejar un JSON a medias."""
    os.makedirs(os.path.dirname(RUTA_CACHE_DESCRIPCIONES), exist_ok=True)
    temporal = RUTA_CACHE_DESCRIPCIONES + ".tmp"
    with open(temporal, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=1)
    os.replace(temporal, RUTA_CACHE_DESCRIPCIONES)


# ---------- Documento de búsqueda ----------
def describir_diapositiva(imagen_png: bytes, texto_pagina: str) -> str:
    """Gemini describe la diapositiva (tema, conceptos, fórmulas en LaTeX, términos del estudiante)."""
    mensaje = mensaje_multimodal(
        f"Texto extraído del PDF:\n{texto_pagina or '(sin texto extraíble)'}",
        diapositiva_b64=base64.b64encode(imagen_png).decode("utf-8"),
    )
    respuesta = obtener_llm_auxiliar().invoke(
        [SYSTEM_PROMPT_DESCRIPCION, mensaje]
    )
    descripcion = texto_de(respuesta).strip()
    if not descripcion:
        raise RuntimeError("Gemini devolvió una descripción vacía.")
    return descripcion


def documento_de_busqueda(
    titulo: str, numero: int, descripcion: str, texto_pagina: str, texto_anterior: str
) -> str:
    """Lo que se indexa en texto (contextual retrieval): contexto + descripción + texto literal."""
    partes = [f"Documento: {titulo} · Diapositiva {numero}"]
    if texto_anterior:
        partes.append(f"Contexto (diapositiva anterior): {texto_anterior[:300]}")
    partes.append(f"Descripción: {descripcion}")
    if texto_pagina:
        partes.append(f"Texto de la diapositiva: {texto_pagina}")
    return "\n".join(partes)


def _texto_limpio(pagina: fitz.Page) -> str:
    return " ".join(pagina.get_text().split())


# ---------- Operaciones sobre la base ----------
def borrar_ids(ids: list[str]) -> None:
    for i in range(0, len(ids), TAMANO_LOTE_BORRADO):
        supabase.table(TABLA).delete().in_("id", ids[i : i + TAMANO_LOTE_BORRADO]).execute()


def procesar_e_inyectar_presentacion_pdf(
    pdf_bytes: bytes,
    nombre_archivo: str,
    filas_existentes: list[dict[str, Any]] | None = None,
    dpi_renderizado: int = 150,
    pausa_llm: float = PAUSA_LLM_SEGUNDOS,
) -> dict[str, Any]:
    """
    Vectoriza todas las páginas del PDF y las sincroniza con Supabase.

    Todo o nada por archivo: si alguna página falla, no se escribe nada y la versión
    anterior queda intacta. Si todo va bien: upsert de las páginas actuales y borrado
    de las filas sobrantes de ese archivo.
    """
    try:
        huella = hash_archivo(pdf_bytes)
        registros: list[dict[str, Any]] = []
        cache = cargar_cache_descripciones()

        with fitz.open(stream=pdf_bytes, filetype="pdf") as documento_pdf:
            total_paginas = len(documento_pdf)
            logger.info("Procesando '%s' (%d páginas) a %d DPI...", nombre_archivo, total_paginas, dpi_renderizado)

            textos = [_texto_limpio(p) for p in documento_pdf]
            primera_linea = next((l for l in documento_pdf[0].get_text().splitlines() if l.strip()), "")
            titulo = f"{nombre_archivo} ({primera_linea.strip()[:80]})" if primera_linea else nombre_archivo

            # Hoja por hoja (no TextSplitter para no romper fórmulas)
            escala = dpi_renderizado / 72.0
            for indice in range(total_paginas):
                numero = indice + 1
                pixmap = documento_pdf[indice].get_pixmap(matrix=fitz.Matrix(escala, escala), alpha=False)
                png = pixmap.tobytes("png")

                vector = generar_embedding_imagen(png)
                if not vector:
                    raise RuntimeError(f"No se pudo vectorizar la diapositiva {numero}.")

                clave = clave_descripcion(huella, numero)
                descripcion = cache.get(clave)
                if descripcion is None:
                    descripcion = describir_diapositiva(png, textos[indice])
                    cache[clave] = descripcion
                    guardar_cache_descripciones(cache)  # se persiste en cada paso: un 429 no pierde lo hecho
                    logger.info("  %s p.%d descrita (%d caracteres).", nombre_archivo, numero, len(descripcion))
                    time.sleep(pausa_llm)  # respeta la cuota de Gemini
                else:
                    logger.info("  %s p.%d: descripción reutilizada de la caché.", nombre_archivo, numero)

                registros.append({
                    "id": id_diapositiva(nombre_archivo, numero),
                    "tipo": TIPO_DIAPOSITIVA,
                    "contenido": documento_de_busqueda(
                        titulo, numero, descripcion, textos[indice], textos[indice - 1] if indice else ""
                    ),
                    "metadata": {
                        "archivo": nombre_archivo,
                        "diapositiva": numero,
                        "hash_archivo": huella,
                        "version_indice": VERSION_INDICE,
                    },
                    "embedding": vector,
                })

        # Vectores de texto en lote: mucho más eficiente en GPU que uno por uno
        for registro, vector_texto in zip(
            registros, generar_embeddings_texto([r["contenido"] for r in registros])
        ):
            registro["embedding_texto"] = vector_texto

        supabase.table(TABLA).upsert(registros).execute()

        # Lo que había de este archivo y ya no corresponde a ninguna página actual
        ids_actuales = {r["id"] for r in registros}
        if filas_existentes is None:
            filas_existentes = (
                supabase.table(TABLA).select("id").eq("metadata->>archivo", nombre_archivo).execute().data or []
            )
        sobrantes = [f["id"] for f in filas_existentes if f["id"] not in ids_actuales]
        borrar_ids(sobrantes)

        logger.info(
            "Éxito: '%s' sincronizado (%d diapositivas, %d filas sobrantes borradas).",
            nombre_archivo, len(registros), len(sobrantes),
        )
        return {"estatus": "ok", "paginas_procesadas": len(registros), "filas_borradas": len(sobrantes)}

    except Exception as e:
        logger.error("Error procesando el PDF %s: %s", nombre_archivo, e)
        return {"estatus": "error", "error": str(e)}


def sincronizar_directorio(
    ruta_carpeta: str = "docs",
    forzar: bool = False,
    borrar_huerfanos: bool = True,
    pausa_llm: float = PAUSA_LLM_SEGUNDOS,
) -> PlanSincronizacion | None:
    """Sincroniza docs/ con Supabase: ingesta lo nuevo o modificado y limpia lo que sobra."""
    if not os.path.isdir(ruta_carpeta):
        logger.error("La carpeta local '%s' no existe.", ruta_carpeta)
        return None

    locales = escanear_directorio(ruta_carpeta)
    remoto = indice_remoto()
    plan = planificar_sincronizacion(locales, remoto, forzar)
    logger.info(
        "Plan: %d a ingestar, %d sin cambios, %d huérfanos.",
        len(plan.ingestar), len(plan.omitir), len(plan.huerfanos),
    )

    for nombre in plan.ingestar:
        with open(locales[nombre].ruta, "rb") as f:
            procesar_e_inyectar_presentacion_pdf(
                f.read(), nombre, filas_existentes=remoto.get(nombre, []), pausa_llm=pausa_llm
            )
        time.sleep(pausa_llm)

    if borrar_huerfanos:
        for nombre in plan.huerfanos:
            borrar_ids([f["id"] for f in remoto[nombre]])
            logger.info("Borrado '%s' de la base: ya no está en '%s'.", nombre, ruta_carpeta)

    return plan


# ---------- Modo vigilancia ----------
def huella_directorio(ruta_carpeta: str) -> dict[str, tuple[float, int]]:
    """Firma barata de la carpeta (fecha de modificación y tamaño de cada PDF), sin leer contenidos."""
    firma: dict[str, tuple[float, int]] = {}
    for nombre in os.listdir(ruta_carpeta):
        if nombre.lower().endswith(".pdf"):
            stat = os.stat(os.path.join(ruta_carpeta, nombre))
            firma[nombre] = (stat.st_mtime, stat.st_size)
    return firma


def vigilar_directorio(ruta_carpeta: str = "docs", intervalo: float = 10.0) -> None:
    """
    Sincroniza al arrancar y luego cada vez que cambia docs/ (Ctrl+C para salir).

    Se usa polling en vez de eventos del sistema de archivos: no añade dependencias y
    funciona igual en Windows y en carpetas de OneDrive. Tras detectar un cambio espera
    a que la carpeta se estabilice un intervalo, para no ingestar un PDF a medio copiar.
    """
    sincronizar_directorio(ruta_carpeta)
    ultima = huella_directorio(ruta_carpeta)
    logger.info("Vigilando '%s' cada %.0f s...", ruta_carpeta, intervalo)

    try:
        while True:
            time.sleep(intervalo)
            actual = huella_directorio(ruta_carpeta)
            if actual == ultima:
                continue
            time.sleep(intervalo)
            estable = huella_directorio(ruta_carpeta)
            if estable != actual:
                continue  # sigue cambiando; se reintenta en la próxima vuelta
            logger.info("Cambios detectados en '%s'. Sincronizando...", ruta_carpeta)
            sincronizar_directorio(ruta_carpeta)
            ultima = estable
    except KeyboardInterrupt:
        logger.info("Vigilancia detenida.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--carpeta", default="docs")
    parser.add_argument("--forzar", action="store_true", help="re-ingesta todo aunque no haya cambios")
    parser.add_argument("--no-borrar-huerfanos", action="store_true", help="no borra de la base los PDFs que ya no están")
    parser.add_argument("--watch", action="store_true", help="vigila la carpeta y sincroniza al detectar cambios")
    parser.add_argument("--intervalo", type=float, default=10.0, help="segundos entre revisiones en modo --watch")
    parser.add_argument("--pausa-llm", type=float, default=PAUSA_LLM_SEGUNDOS,
                        help="segundos entre descripciones de Gemini (cuota del plan gratuito)")
    args = parser.parse_args()

    print("Iniciando Motor de Ingesta Local...")
    if args.watch:
        vigilar_directorio(args.carpeta, args.intervalo)
    else:
        sincronizar_directorio(
            args.carpeta, forzar=args.forzar, borrar_huerfanos=not args.no_borrar_huerfanos,
            pausa_llm=args.pausa_llm,
        )
