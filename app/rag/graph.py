"""
Orquestador del tutor con LangGraph.

Flujo:
    START ─▶ enrutar_entrada ─▶ recuperar | END (sin entrada)
    recuperar ─▶ (hay documentos?) ─▶ evaluar_relevancia | sin_contexto
    evaluar_relevancia ─▶ generar_respuesta | siguiente_candidata ─▶ evaluar_relevancia | sin_contexto
    generar_respuesta ─▶ juzgar_respuesta ─▶ END | generar_respuesta (reintento con feedback)

- recuperar enruta por modalidad (texto → híbrido bge-m3 + BM25; imagen → CLIP; ambos → RRF).
- evaluar_relevancia detecta fallos de RECUPERACIÓN antes de generar (estilo CRAG).
- juzgar_respuesta (LLM-as-judge) detecta fallos de GENERACIÓN y devuelve feedback
  para que el siguiente intento corrija la respuesta (generate → evaluate → refine).
"""

import logging
from typing import Any, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from app.models.schemas import RelevanciaDiapositiva, VeredictoJuez
from app.rag.diapositivas import renderizar_diapositiva_base64
from app.rag.llm import mensaje_multimodal, obtener_llm, obtener_llm_auxiliar, texto_de
from app.rag.prompts import SYSTEM_PROMPT_GRADER, SYSTEM_PROMPT_JUEZ, SYSTEM_PROMPT_TUTOR
from app.rag.retrieval import MODO_POR_DEFECTO, ModoRecuperacion, recuperar as recuperar_candidatas

logger = logging.getLogger(__name__)

# Guarda contra bucles: número máximo de generaciones por consulta.
MAX_INTENTOS = 2

RESPUESTA_SIN_CONTEXTO = (
    "No encontré en las diapositivas del curso información suficiente para responder "
    "esta pregunta. Prueba a reformularla o a indicar el tema concreto."
)


#Definicion del estado global
class EstadoTutor(TypedDict, total=False):
    # Entrada
    consulta: Optional[str]
    image_base64: Optional[str]
    top_k: int
    modo_recuperacion: ModoRecuperacion
    # Recuperación
    documentos: list[dict[str, Any]]
    indice_doc: int                # candidata que se está evaluando dentro de documentos
    diapositiva_b64: str
    relevante: bool
    motivo_relevancia: str
    # Generación y evaluación
    respuesta: str
    veredicto: dict[str, Any]
    feedback: str
    intentos: int


def _texto_consulta(state: EstadoTutor) -> str:
    consulta = state.get("consulta")
    return consulta if consulta else "Explica el contenido de la imagen que envié."


#Nodos
def recuperar(state: EstadoTutor) -> dict[str, Any]:
    documentos = recuperar_candidatas(
        consulta=state.get("consulta"),
        image_base64=state.get("image_base64"),
        top_k=state.get("top_k", 3),
        modo=state.get("modo_recuperacion", MODO_POR_DEFECTO),
    )
    logger.info("Recuperadas %d diapositivas únicas.", len(documentos))
    return {"documentos": documentos, "indice_doc": 0}


def evaluar_relevancia(state: EstadoTutor) -> dict[str, Any]:
    """Gemini mira la diapositiva candidata y decide si sirve para la consulta."""
    doc = state["documentos"][state["indice_doc"]]
    meta = doc.get("metadata") or {}
    diapositiva_b64 = renderizar_diapositiva_base64(meta["archivo"], int(meta["diapositiva"]))

    grader = obtener_llm_auxiliar().with_structured_output(RelevanciaDiapositiva)
    mensaje = mensaje_multimodal(
        f"Consulta del estudiante: {_texto_consulta(state)}",
        diapositiva_b64=diapositiva_b64,
        imagen_estudiante_b64=state.get("image_base64"),
    )
    resultado: RelevanciaDiapositiva = grader.invoke([SYSTEM_PROMPT_GRADER, mensaje])
    logger.info(
        "Relevancia %s p.%s: %s (%s)",
        meta.get("archivo"), meta.get("diapositiva"), resultado.relevante, resultado.motivo,
    )
    return {
        "diapositiva_b64": diapositiva_b64,
        "relevante": resultado.relevante,
        "motivo_relevancia": resultado.motivo,
    }


def siguiente_candidata(state: EstadoTutor) -> dict[str, Any]:
    return {"indice_doc": state["indice_doc"] + 1}


def sin_contexto(state: EstadoTutor) -> dict[str, Any]:
    return {"respuesta": RESPUESTA_SIN_CONTEXTO, "relevante": False}


def generar_respuesta(state: EstadoTutor) -> dict[str, Any]:
    """Genera la respuesta; en un reintento incluye la respuesta anterior y el feedback del juez."""
    texto = f"Consulta del estudiante: {_texto_consulta(state)}"
    if state.get("feedback"):
        texto += (
            "\n\nTu respuesta anterior fue rechazada por el revisor.\n"
            f"Respuesta anterior:\n{state['respuesta']}\n\n"
            f"Correcciones que debes aplicar: {state['feedback']}"
        )

    mensaje = mensaje_multimodal(
        texto,
        diapositiva_b64=state["diapositiva_b64"],
        imagen_estudiante_b64=state.get("image_base64"),
    )
    respuesta = obtener_llm(temperatura=0.2).invoke([SYSTEM_PROMPT_TUTOR, mensaje])
    return {"respuesta": texto_de(respuesta), "intentos": state.get("intentos", 0) + 1}


def juzgar_respuesta(state: EstadoTutor) -> dict[str, Any]:
    """LLM-as-judge de fidelidad. Temperatura 0 para que el veredicto sea estable."""
    juez = obtener_llm_auxiliar().with_structured_output(VeredictoJuez)
    mensaje = mensaje_multimodal(
        f"Consulta del estudiante: {_texto_consulta(state)}\n\n"
        f"Respuesta del tutor:\n{state['respuesta']}",
        diapositiva_b64=state["diapositiva_b64"],
        imagen_estudiante_b64=state.get("image_base64"),
    )
    veredicto: VeredictoJuez = juez.invoke([SYSTEM_PROMPT_JUEZ, mensaje])
    logger.info(
        "Juez intento %d: aprobada=%s puntaje=%d", state["intentos"], veredicto.aprobada, veredicto.puntaje
    )
    return {"veredicto": veredicto.model_dump(), "feedback": veredicto.motivo}


#Bordes condicionales
def enrutar_entrada(state: EstadoTutor) -> str:
    """Sin texto ni imagen no hay nada que recuperar."""
    if state.get("image_base64") or state.get("consulta"):
        return "recuperar"
    return "sin_entrada"


def hay_documentos(state: EstadoTutor) -> str:
    return "evaluar_relevancia" if state.get("documentos") else "sin_contexto"


def decidir_relevancia(state: EstadoTutor) -> str:
    if state["relevante"]:
        return "generar_respuesta"
    if state["indice_doc"] + 1 < len(state["documentos"]):
        return "siguiente_candidata"
    return "sin_contexto"


def decidir_tras_juicio(state: EstadoTutor) -> str:
    if state["veredicto"]["aprobada"] or state.get("intentos", 0) >= MAX_INTENTOS:
        return "fin"
    return "regenerar"


#Construccion del grafo
builder = StateGraph(EstadoTutor)

builder.add_node("recuperar", recuperar)
builder.add_node("evaluar_relevancia", evaluar_relevancia)
builder.add_node("siguiente_candidata", siguiente_candidata)
builder.add_node("sin_contexto", sin_contexto)
builder.add_node("generar_respuesta", generar_respuesta)
builder.add_node("juzgar_respuesta", juzgar_respuesta)

builder.add_conditional_edges(START, enrutar_entrada, {
    "recuperar": "recuperar",
    "sin_entrada": END,
})
builder.add_conditional_edges("recuperar", hay_documentos, {
    "evaluar_relevancia": "evaluar_relevancia",
    "sin_contexto": "sin_contexto",
})
builder.add_conditional_edges("evaluar_relevancia", decidir_relevancia, {
    "generar_respuesta": "generar_respuesta",
    "siguiente_candidata": "siguiente_candidata",
    "sin_contexto": "sin_contexto",
})
builder.add_edge("siguiente_candidata", "evaluar_relevancia")
builder.add_edge("sin_contexto", END)
builder.add_edge("generar_respuesta", "juzgar_respuesta")
builder.add_conditional_edges("juzgar_respuesta", decidir_tras_juicio, {
    "regenerar": "generar_respuesta",
    "fin": END,
})

#Compilamos el grafo
tutor_graph = builder.compile()
