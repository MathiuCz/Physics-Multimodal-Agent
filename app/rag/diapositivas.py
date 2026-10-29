"""
Render de diapositivas locales: abre un PDF de docs/ y convierte una página a PNG en Base64.
"""

import base64
import os

import fitz

DPI_POR_DEFECTO = 150


def renderizar_diapositiva_base64(
    nombre_archivo: str,
    numero_diapositiva: int,
    ruta_carpeta: str = "docs",
    dpi: int = DPI_POR_DEFECTO,
) -> str:
    """Renderiza la diapositiva (numeración desde 1) y la devuelve como PNG en Base64."""
    ruta_pdf = os.path.join(ruta_carpeta, nombre_archivo)
    if not os.path.isfile(ruta_pdf):
        raise FileNotFoundError(f"No existe el PDF '{ruta_pdf}'.")

    # El context manager cierra el documento aunque falle el render.
    with fitz.open(ruta_pdf) as doc:
        if not 1 <= numero_diapositiva <= len(doc):
            raise ValueError(
                f"'{nombre_archivo}' tiene {len(doc)} páginas; se pidió la {numero_diapositiva}."
            )
        escala = dpi / 72
        pixmap = doc[numero_diapositiva - 1].get_pixmap(
            matrix=fitz.Matrix(escala, escala), alpha=False
        )
        png = pixmap.tobytes("png")

    return base64.b64encode(png).decode("utf-8")


def contar_diapositivas(nombre_archivo: str, ruta_carpeta: str = "docs") -> int:
    """Número de páginas del PDF."""
    with fitz.open(os.path.join(ruta_carpeta, nombre_archivo)) as doc:
        return len(doc)
