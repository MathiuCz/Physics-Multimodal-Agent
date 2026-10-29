import logging

from fastapi import APIRouter, HTTPException

from app.models.schemas import BusquedaRequest, BusquedaResponse
from app.rag.service import responder_consulta

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/", response_model=BusquedaResponse)
def buscar_documentos(payload: BusquedaRequest) -> BusquedaResponse:
    """Recibe una consulta (texto y/o imagen) y devuelve la respuesta del tutor ya evaluada."""
    try:
        resultado = responder_consulta(
            consulta=payload.consulta,
            image_base64=payload.image_base64,
            top_k=payload.top_k,
        )
        return BusquedaResponse(**resultado)

    except ValueError as e:  # entrada inválida (sin texto ni imagen, imagen corrupta...)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("Error ejecutando el tutor")
        raise HTTPException(status_code=500, detail="Error interno al procesar la consulta.")
