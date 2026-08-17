from typing import Dict, Optional, Any

from pydantic import BaseModel

#Modelo para cuando se inserta texto 
class DocumentRequest(BaseModel):
    texto: str
    metadatos: Dict = {}

    image_base64: Optional[str] = None
    url_image: Optional[str] = None

#Modelo para cuando se busca o genera embedding
class BusquedaRequest(BaseModel):
    consulta: str
    sesion_id: str = ""
    top_k: int = 3
    image_base64: Optional[str] = None

#Modelo para cuando se envia la respuesta 
class BusquedaResponse(BaseModel):
    respuesta: str
    documentos_recuperados: list[Dict[str, Any]]
