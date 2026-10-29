from typing import Dict, Optional, Any

from pydantic import BaseModel, Field

#Modelo para cuando se inserta texto
class DocumentRequest(BaseModel):
    texto: str
    metadatos: Dict = {}

    image_base64: Optional[str] = None
    url_image: Optional[str] = None

#Modelo para cuando se busca o genera embedding
class BusquedaRequest(BaseModel):
    consulta: Optional[str] = None
    sesion_id: str = ""
    top_k: int = Field(default=3, ge=1, le=10)
    image_base64: Optional[str] = None


#Salidas estructuradas que devuelven los LLM evaluadores del grafo
class RelevanciaDiapositiva(BaseModel):
    relevante: bool = Field(description="True si la diapositiva contiene información para responder la consulta")
    motivo: str = Field(description="Justificación breve de la decisión")


class VeredictoJuez(BaseModel):
    aprobada: bool = Field(description="True si la respuesta es fiel a la diapositiva y responde la consulta")
    puntaje: int = Field(ge=1, le=5, description="Fidelidad al contexto: 1 inventada, 5 totalmente sustentada")
    motivo: str = Field(description="Instrucciones accionables para corregir la respuesta, o por qué se aprueba")


class DiapositivaUsada(BaseModel):
    archivo: str
    diapositiva: int


#Modelo para cuando se envia la respuesta
class BusquedaResponse(BaseModel):
    respuesta: str
    documentos_recuperados: list[Dict[str, Any]]
    diapositiva_usada: Optional[DiapositivaUsada] = None
    veredicto: Optional[VeredictoJuez] = None
    intentos: int = 0
