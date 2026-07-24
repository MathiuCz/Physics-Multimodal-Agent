import io
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"

# Modelo multimodal y multilingue
model = SentenceTransformer("clip-ViT-B-32-multilingual-v1", device=device)


def generar_embedding(texto: str) -> list[float]:
    """Genera el embedding de un texto"""
    embeddings = model.encode(texto)
    return embeddings.tolist()


def generar_embedding_imagen(imagen_bytes: bytes) -> list[float]:
    """Genera el embedding de una imagen a partir de sus bytes"""
    imagen = Image.open(io.BytesIO(imagen_bytes))
    embeddings = model.encode(imagen)
    return embeddings.tolist()
