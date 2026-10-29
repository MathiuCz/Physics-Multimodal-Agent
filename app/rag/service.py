"""
Servicio del tutor: punto de entrada único al pipeline RAG.

El router HTTP (y cualquier script local) llama a responder_consulta; la orquestación
vive en el grafo (app/rag/graph.py). Así hay una sola implementación del flujo.
"""

import logging
from typing import Any, Optional

from app.rag.graph import MAX_INTENTOS, tutor_graph
from app.rag.retrieval import MODO_POR_DEFECTO, ModoRecuperacion

logger = logging.getLogger(__name__)


def responder_consulta(
    consulta: Optional[str] = None,
    image_base64: Optional[str] = None,
    top_k: int = 3,
    modo: ModoRecuperacion = MODO_POR_DEFECTO,
) -> dict[str, Any]:
    """Ejecuta el grafo y devuelve un dict con la forma de BusquedaResponse."""
    if not consulta and not image_base64:
        raise ValueError("Debes enviar un texto o una imagen.")

    estado = tutor_graph.invoke(
        {"consulta": consulta, "image_base64": image_base64, "top_k": top_k, "modo_recuperacion": modo},
        # Margen holgado: cada candidata y cada reintento consumen pasos del grafo.
        config={"recursion_limit": 10 + 2 * top_k + 2 * MAX_INTENTOS},
    )

    documentos = estado.get("documentos", [])
    diapositiva_usada = None
    if estado.get("relevante") and documentos:
        meta = documentos[estado["indice_doc"]].get("metadata") or {}
        diapositiva_usada = {"archivo": meta.get("archivo"), "diapositiva": meta.get("diapositiva")}

    return {
        "respuesta": estado.get("respuesta", ""),
        "documentos_recuperados": documentos,
        "diapositiva_usada": diapositiva_usada,
        "veredicto": estado.get("veredicto"),
        "intentos": estado.get("intentos", 0),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print("Iniciando Prueba del Motor Semántico...\n")

    # 1. Escribe aquí cualquier pregunta sobre tus diapositivas:
    pregunta = "¿Que es un hamiltoneano?"

    resultado = responder_consulta(pregunta)

    print("=== RESPUESTA DE GEMINI ===")
    print(resultado["respuesta"])
    print("\n=== DIAPOSITIVA USADA ===")
    print(resultado["diapositiva_usada"])
    print("\n=== VEREDICTO DEL JUEZ ===")
    print(resultado["veredicto"], f"(intentos: {resultado['intentos']})")
