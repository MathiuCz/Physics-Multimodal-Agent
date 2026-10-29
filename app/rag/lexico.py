"""
Búsqueda léxica BM25 en memoria sobre el campo `contenido` de las diapositivas.

Con decenas o cientos de diapositivas, el índice completo cabe en memoria y se calcula en
milisegundos. A escala de millones se usaría el motor de la base (p. ej. ParadeDB/pg_search)
o un motor dedicado (Elasticsearch/OpenSearch). El `ts_rank` nativo de Postgres no sirve
como sustituto: no usa IDF, que es la mitad de BM25.

BM25 combina tres ideas:
- IDF: un término raro ("hamiltoniano") pesa más que uno común ("función").
- Frecuencia con saturación (k1): repetir un término suma, pero cada vez menos.
- Normalización por longitud (b): un documento largo no gana solo por tener más palabras.
"""

import logging
import math
import re
import time
import unicodedata
from collections import Counter
from typing import Any

from app.core.supabase_client import client

logger = logging.getLogger(__name__)

# Tiempo de vida del índice en el proceso de la API: la ingesta corre en otro proceso,
# así que el índice se reconstruye periódicamente para ver los cambios.
TTL_INDICE_SEGUNDOS = 60.0

STOPWORDS = {
    "a", "al", "algo", "como", "con", "cual", "cuales", "cuando", "de", "del", "donde", "e",
    "el", "ella", "en", "entre", "es", "esa", "ese", "eso", "esta", "este", "esto", "estos",
    "fue", "ha", "hay", "la", "las", "le", "lo", "los", "mas", "me", "mi", "muy", "no", "o",
    "para", "pero", "por", "porque", "que", "se", "segun", "ser", "si", "sin", "sobre", "son",
    "su", "sus", "tambien", "te", "tiene", "un", "una", "unas", "uno", "unos", "y", "ya",
    "explica", "explicame", "significa", "dime", "diapositiva",
}


def _raiz(token: str) -> str:
    """Stemming mínimo de plurales: ecuaciones→ecuacion, clases→clase, operadores→operador."""
    if len(token) > 4 and token.endswith("es") and token[-3] in "nrldz":
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def tokenizar(texto: str) -> list[str]:
    """Minúsculas, sin tildes, sin stopwords y con plurales reducidos."""
    texto = unicodedata.normalize("NFKD", texto.lower())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return [_raiz(t) for t in re.findall(r"[a-z0-9]+", texto) if t not in STOPWORDS and len(t) > 1]


class IndiceBM25:
    def __init__(self, documentos: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        self.documentos = documentos
        self.k1, self.b = k1, b
        self.frecuencias = [Counter(tokenizar(d.get("contenido") or "")) for d in documentos]
        self.longitudes = [sum(f.values()) for f in self.frecuencias]
        self.longitud_media = (sum(self.longitudes) / len(self.longitudes)) if documentos else 0.0

        n = len(documentos)
        df: Counter[str] = Counter()
        for f in self.frecuencias:
            df.update(f.keys())
        # IDF de BM25 (variante con +1 para que nunca sea negativa)
        self.idf = {t: math.log(1 + (n - nt + 0.5) / (nt + 0.5)) for t, nt in df.items()}

    def buscar(self, consulta: str, top_k: int) -> list[dict[str, Any]]:
        terminos = tokenizar(consulta)
        resultados: list[tuple[float, int]] = []
        for i, frec in enumerate(self.frecuencias):
            puntaje = 0.0
            normalizacion = 1 - self.b + self.b * self.longitudes[i] / (self.longitud_media or 1)
            for t in terminos:
                f = frec.get(t)
                if f:
                    puntaje += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * normalizacion)
            if puntaje > 0:
                resultados.append((puntaje, i))

        resultados.sort(reverse=True)
        return [{**self.documentos[i], "puntaje_bm25": p} for p, i in resultados[:top_k]]


_cache: dict[str, Any] = {"indice": None, "creado": 0.0}


def indice_bm25() -> IndiceBM25:
    """Índice construido desde Supabase y reutilizado durante TTL_INDICE_SEGUNDOS."""
    ahora = time.monotonic()
    if _cache["indice"] is None or ahora - _cache["creado"] > TTL_INDICE_SEGUNDOS:
        filas = (
            client.table("documentos")
            .select("id, tipo, contenido, metadata")
            .eq("tipo", "imagen")
            .execute()
            .data
            or []
        )
        _cache["indice"] = IndiceBM25(filas)
        _cache["creado"] = ahora
        logger.info("Índice BM25 reconstruido con %d diapositivas.", len(filas))
    return _cache["indice"]
