import io
import logging
from functools import lru_cache

import torch
from PIL import Image, UnidentifiedImageError
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Detectar hardware (GPU o CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Cargando modelos de Embeddings en dispositivo: {device}")

# MODELO DE TEXTO CLIP (Multilingüe): mismo espacio que las imágenes
embeddings_texto = HuggingFaceEmbeddings(
    model_name="clip-ViT-B-32-multilingual-v1",
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True}
)

# MODELO DE IMAGEN CLIP
modelo_imagen = SentenceTransformer("clip-ViT-B-32", device=device)

# MODELO DE TEXTO PARA BÚSQUEDA SEMÁNTICA (índice de texto de las diapositivas).
# Se carga en diferido: solo lo pagan los procesos que lo usan (ingesta y consultas de texto).
MODELO_TEXTO = "BAAI/bge-m3"
DIMENSION_TEXTO = 1024


def generar_embedding(texto: str) -> list[float]:
    """Vector CLIP de una consulta de texto (comparable con los vectores de imagen)."""
    return embeddings_texto.embed_query(texto)


def generar_embedding_imagen(imagen_bytes: bytes) -> list[float]:
    """Genera vector a partir de los bytes de una diapositiva renderizada."""
    try:
        imagen = Image.open(io.BytesIO(imagen_bytes))
        resultado = modelo_imagen.encode(imagen)
        return resultado.tolist()

    except UnidentifiedImageError:
        logger.error("Los bytes proporcionados no son una imagen válida.")
        return []
    except Exception as e:
        logger.error(f"Error procesando la imagen: {e}")
        return []


@lru_cache(maxsize=1)
def _modelo_texto() -> SentenceTransformer:
    logger.info("Cargando %s en %s...", MODELO_TEXTO, device)
    modelo = SentenceTransformer(MODELO_TEXTO, device=device)
    if device == "cuda":
        modelo.half()  # fp16: la mitad de VRAM, sin pérdida apreciable para recuperación
    return modelo


def generar_embeddings_texto(textos: list[str]) -> list[list[float]]:
    """
    Vectores bge-m3 normalizados (1024-d) para documentos o consultas.

    bge-m3 no necesita prefijos de consulta/pasaje (a diferencia de e5), así que la misma
    función sirve para indexar y para buscar.
    """
    vectores = _modelo_texto().encode(
        textos, batch_size=8, normalize_embeddings=True, convert_to_numpy=True
    )
    return vectores.astype("float32").tolist()
