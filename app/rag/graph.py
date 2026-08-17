"""
Archivo que contiene todo los grafos para orquestarlo con LangGraph

"""

from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
import base64

#Importaciones de las funciones
from app.rag.embedding import generar_embedding, generar_embedding_imagen
from app.core.supabase_client import client as supabase

#Definicion del estado global
class EstadoMotorSemantico(TypedDict):
    consulta: Optional[str] #estados
    image_base64: Optional[str]
    top_k: int
    vector: List[float]
    resultados: List[Dict[str, Any]]
    respuesta_final: str

#Nodos Funciones que envuelven a las funiones que ya tenemos en el proyecto
def nodo_vectorizar_texto(state: EstadoMotorSemantico) -> Dict[str, Any]:
    """Nodo que invoca a generar_embedding en app/core/embedding.py"""
    vector_texto = generar_embedding(state["consulta"])
    return {"vector": vector_texto}

def nodo_vectorizar_imagen(state: EstadoMotorSemantico) -> Dict[str, Any]:
    """Nodo que decodifica base64 e invoca a generar_embedding_imagen"""
    datos_base64 = state["image_base64"]
    if "," in datos_base64:
        datos_base64 = datos_base64.split(",")[1]
    
    bytes_imagen = base64.b64decode(datos_base64)
    vector_imagen = generar_embedding_imagen(bytes_imagen)
    return {"vector": vector_imagen}

def nodo_buscar_supabase(state: EstadoMotorSemantico) -> Dict[str, Any]:
    """Nodo que consulta la RPC buscar_similares en Supabase"""
    resultado_rpc = supabase.rpc(
        "buscar_similares",
        {
            "query": state["vector"],
            "top_k": state["top_k"]
        }
    ).execute()
    
    documentos = resultado_rpc.data if resultado_rpc.data else []
    return {"resultados": documentos}

def nodo_sintetizar_respuesta(state: EstadoMotorSemantico) -> Dict[str, Any]:
    """Nodo que formatea la salida final"""
    total = len(state["resultados"])
    return {"respuesta_final": f"Se encontraron {total} documentos o imágenes similares."}

#Borde condicional (Enrutador de entrada)
def enrutar_entrada(state: EstadoMotorSemantico) -> str:
    """Decide el nodo inicial según si el usuario envió texto o imagen"""
    if state.get("image_base64"):
        return "vectorizar_imagen"
    elif state.get("consulta"):
        return "vectorizar_texto"
    else:
        return "sin_entrada"

#Construccion del grafo

biulder = StateGraph(EstadoMotorSemantico)

#Añadimos los nodos

biulder.add_node("vectorizar_texto", nodo_vectorizar_texto)
biulder.add_node("vectorizar_imagen", nodo_vectorizar_imagen)
biulder.add_node("buscar_supabase", nodo_buscar_supabase)
biulder.add_node("sintetizar_respuesta", nodo_sintetizar_respuesta)

#Añadimos el borde condicional

biulder.add_conditional_edges(
    START,
    enrutar_entrada,
    {
        "vectorizar_texto": "vectorizar_texto",
        "vectorizar_imagen": "vectorizar_imagen",
        "sin_entrada": END
    }
)

# Definimos los flujos entre nodos
biulder.add_edge("vectorizar_texto", "buscar_supabase")
biulder.add_edge("vectorizar_imagen", "buscar_supabase")
biulder.add_edge("buscar_supabase", "sintetizar_respuesta")
biulder.add_edge("sintetizar_respuesta", END)


#Compilamos el grafo
motor_semantico_graph = biulder.compile()