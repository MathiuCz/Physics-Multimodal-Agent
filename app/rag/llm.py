"""
Acceso a Gemini: un único punto donde se configuran modelo, temperatura y límites,
y helpers para construir mensajes multimodales.
"""

from functools import lru_cache
from typing import Any, Optional

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import GEMINI_API_KEY

# Enrutado de modelos (model routing): cada tarea usa el modelo más barato que la resuelve bien.
# La cuota gratuita es POR MODELO: Flash ~20 peticiones/día, Flash-Lite ~500/día (ver ai.dev/rate-limit).
MODELO_GENERACION = "gemini-3.5-flash"       # respuesta al estudiante: la única donde la calidad se nota directamente
MODELO_AUXILIAR = "gemini-3.5-flash-lite"    # grader, juez, descripciones de ingesta y dataset sintético


@lru_cache(maxsize=None)
def obtener_llm(
    temperatura: float = 0.2,
    max_tokens_respuesta: int = 800,
    presupuesto_razonamiento: int = 512,
    modelo: str = MODELO_GENERACION,
) -> ChatGoogleGenerativeAI:
    """
    Devuelve un cliente de Gemini reutilizable (uno por combinación de parámetros).

    Gemini 3.x "piensa" antes de responder y esos tokens cuentan dentro del límite de
    salida: con un único límite de 800, el razonamiento se lo comía casi entero y la
    respuesta salía cortada (finish_reason=MAX_TOKENS). Por eso se separan ambos
    presupuestos y el límite total es su suma.
    """
    if not GEMINI_API_KEY:
        raise RuntimeError("Falta GEMINI_API_KEY en el .env.")
    return ChatGoogleGenerativeAI(
        model=modelo,
        temperature=temperatura,
        thinking_budget=presupuesto_razonamiento,
        max_output_tokens=max_tokens_respuesta + presupuesto_razonamiento,
        google_api_key=GEMINI_API_KEY,
    )


def obtener_llm_auxiliar(temperatura: float = 0.0, max_tokens_respuesta: int = 600) -> ChatGoogleGenerativeAI:
    """Modelo de tareas auxiliares (evaluar, describir, generar datos): barato y con cuota amplia."""
    return obtener_llm(temperatura, max_tokens_respuesta, 512, MODELO_AUXILIAR)


def bloque_imagen(imagen_b64: str, mime: str = "image/png") -> dict[str, Any]:
    """Bloque de contenido image_url con data URI."""
    if imagen_b64.startswith("data:"):
        return {"type": "image_url", "image_url": imagen_b64}
    return {"type": "image_url", "image_url": f"data:{mime};base64,{imagen_b64}"}


def mensaje_multimodal(
    texto: str,
    diapositiva_b64: Optional[str] = None,
    imagen_estudiante_b64: Optional[str] = None,
) -> HumanMessage:
    """HumanMessage con texto, la imagen del estudiante (si la hay) y la diapositiva (si la hay)."""
    contenido: list[dict[str, Any]] = [{"type": "text", "text": texto}]
    if imagen_estudiante_b64:
        contenido.append({"type": "text", "text": "Imagen enviada por el estudiante:"})
        contenido.append(bloque_imagen(imagen_estudiante_b64))
    if diapositiva_b64:
        contenido.append({"type": "text", "text": "Diapositiva recuperada del curso:"})
        contenido.append(bloque_imagen(diapositiva_b64))
    return HumanMessage(content=contenido)


def texto_de(mensaje: BaseMessage) -> str:
    """Texto plano de la respuesta, sin bloques de metadatos ni firmas."""
    return str(mensaje.text)
