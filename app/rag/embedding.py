import io
import torch
import logging
from PIL import Image, UnidentifiedImageError
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

# Detectar hardware (GPU o CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Cargando modelos de Embeddings en dispositivo: {device}")

# MODELO DE TEXTO (Multilingüe)
embeddings_texto = HuggingFaceEmbeddings(
    model_name="clip-ViT-B-32-multilingual-v1",
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True}
)

# MODELO DE IMAGEN
modelo_imagen = SentenceTransformer("clip-ViT-B-32", device=device)

def generar_embedding(texto: str) -> list[float]:
    """Genera vector para la consulta de texto en español."""
    return embeddings_texto.embed_query(texto)


def generar_embedding_imagen(imagen_bytes: bytes) -> list[float]:
    """Genera vector a partir de los bytes de una diapositiva renderizada."""
    try:
        imagen = Image.open(io.BytesIO(imagen_bytes))
        resultado = modelo_imagen.encode(imagen)
        return resultado.tolist()
        
    except UnidentifiedImageError:
        logging.error("Los bytes proporcionados no son una imagen válida.")
        return []
    except Exception as e:
        logging.error(f"Error procesando la imagen: {e}")
        return []