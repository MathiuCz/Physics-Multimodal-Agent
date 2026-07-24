from typing import Dict, Optional

from pydantic import BaseModel

class DocumentRequest(BaseModel):
    texto: str
    metadatos: Dict = {}

    image_base64: Optional[str] = None
    url_image: Optional[str] = None

class BusquedaRequest(BaseModel):
    consulta: str
    sesion_id: str = ""
    top_k: int = 3

    image_base64: Optional[str] = None

    
    