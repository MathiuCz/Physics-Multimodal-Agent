from app.core import supabase
from fastapi import APIRouter, HTTPException

from app.core.embedding import generar_embedding
from app.core.supabase_client import client
from app.models.schemas import BusquedaRequest, BusquedaResponse


router = APIRouter()

@router .post("/")
def buscar_documentos(payload: BusquedaRequest):#payload contiene los datos de la peticion 
    #BusquedaRequest es el esquema que define los datos que se esperan en la peticion 
    try:
        vector = generar_embedding(payload.consulta) #genera el embedding de la consulta

        resultado = supabase.rpc(
            "buscar_similares",
            {
                "query": vector, "top_k": payload.top_k 
            }
        ).execute()

        return {"resultados" : resultado.data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    