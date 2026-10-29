"""
Genera un dataset sintético para evaluar el recuperador.

Para cada diapositiva de docs/, Gemini (modelo auxiliar) escribe preguntas que esa diapositiva
responde. La etiqueta sale gratis: la diapositiva relevante es la que originó la pregunta.

Sesgo conocido: las preguntas sintéticas tienden a copiar el vocabulario de la fuente y
las métricas salen optimistas. Se mitiga pidiendo paráfrasis y mezclando con preguntas
escritas a mano (agrégalas al JSON con "origen": "manual").

Independencia del índice: se genera viendo SOLO la imagen de la diapositiva, en llamadas
separadas de las descripciones de la ingesta. Si preguntas y descripciones salieran de la
misma llamada, compartirían vocabulario y la evaluación tendría fuga de datos.

Reanudable: guarda tras cada diapositiva y, al relanzarlo, salta las que ya tienen preguntas.

Uso (desde la raíz):
    uv run python -m app.eval.generar_dataset --preguntas-por-diapositiva 2
"""

import argparse
import json
import logging
import os
import time

from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field

from app.rag.diapositivas import contar_diapositivas, renderizar_diapositiva_base64
from app.rag.llm import mensaje_multimodal, obtener_llm_auxiliar

logger = logging.getLogger(__name__)

RUTA_SALIDA = os.path.join("app", "eval", "datasets", "golden_sintetico.json")

PROMPT_GENERADOR = SystemMessage(
    content=(
        "Eres un estudiante universitario de física preparando un examen.\n"
        "Mira la diapositiva y escribe preguntas que esa diapositiva permite responder.\n"
        "Reglas:\n"
        "- Escribe como lo haría un estudiante real, en español y con tus propias palabras: "
        "NO copies frases ni títulos literales de la diapositiva.\n"
        "- Mezcla preguntas conceptuales ('¿qué significa...?', '¿por qué...?') y sobre fórmulas.\n"
        "- Si la diapositiva es una portada, índice, agradecimiento o no tiene contenido "
        "de física, devuelve una lista vacía."
    )
)


class PreguntasGeneradas(BaseModel):
    preguntas: list[str] = Field(description="Preguntas que la diapositiva responde; vacía si no tiene contenido")


def _cargar(ruta: str) -> list[dict]:
    try:
        with open(ruta, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def _guardar(ruta: str, datos: list[dict]) -> None:
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    temporal = ruta + ".tmp"
    with open(temporal, "w", encoding="utf-8") as f:
        json.dump(datos, f, ensure_ascii=False, indent=2)
    os.replace(temporal, ruta)


def generar(carpeta: str, preguntas_por_diapositiva: int, pausa: float, salida: str) -> list[dict]:
    generador = obtener_llm_auxiliar(temperatura=0.7).with_structured_output(PreguntasGeneradas)
    dataset = _cargar(salida)
    # Diapositivas ya resueltas: tienen preguntas o se marcaron como "sin contenido".
    hechas = {tuple(c["relevantes"][0]) for c in dataset}
    hechas |= {tuple(c["diapositiva"]) for c in dataset if c.get("origen") == "sin_contenido"}
    fallos = 0

    pdfs = sorted(f for f in os.listdir(carpeta) if f.lower().endswith(".pdf"))
    for archivo in pdfs:
        for numero in range(1, contar_diapositivas(archivo, carpeta) + 1):
            if (archivo, numero) in hechas:
                continue
            imagen = renderizar_diapositiva_base64(archivo, numero, carpeta)
            mensaje = mensaje_multimodal(
                f"Escribe hasta {preguntas_por_diapositiva} preguntas.", diapositiva_b64=imagen
            )
            try:
                resultado: PreguntasGeneradas = generador.invoke([PROMPT_GENERADOR, mensaje])
            except Exception as e:
                fallos += 1
                logger.warning("Fallo en %s p.%d (se reintentará al relanzar): %s", archivo, numero, str(e)[:200])
                time.sleep(pausa)
                continue

            preguntas = resultado.preguntas[:preguntas_por_diapositiva]
            for pregunta in preguntas:
                dataset.append({"pregunta": pregunta, "relevantes": [[archivo, numero]], "origen": "sintetico"})
            if not preguntas:
                # Marcador para no volver a preguntar por portadas; el evaluador lo ignora.
                dataset.append({"diapositiva": [archivo, numero], "origen": "sin_contenido"})
            _guardar(salida, dataset)
            logger.info("%s p.%d → %d preguntas", archivo, numero, len(preguntas))
            time.sleep(pausa)  # respeta la cuota por minuto de la API

    if fallos:
        logger.warning("%d diapositivas fallaron: relanza el comando para completarlas.", fallos)
    return dataset


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--carpeta", default="docs")
    parser.add_argument("--preguntas-por-diapositiva", type=int, default=2)
    parser.add_argument("--pausa", type=float, default=6.0, help="segundos entre llamadas a Gemini")
    parser.add_argument("--salida", default=RUTA_SALIDA)
    args = parser.parse_args()

    datos = generar(args.carpeta, args.preguntas_por_diapositiva, args.pausa, args.salida)
    preguntas = [c for c in datos if "pregunta" in c]
    print(f"{len(preguntas)} preguntas en {args.salida}")
