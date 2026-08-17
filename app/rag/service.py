import langchain_google_genai
from langchain_core.tracers import langchain
import os
import io
import fitz
import base64
import logging
from typing import List, Dict, Any, Optional

from app.rag.embedding import generar_embedding, generar_embedding_imagen
from app.rag.prompts import SYSTEM_PROMPT_TUTOR
from app.core.supabase_client import client
from app.core.config import GEMINI_API_KEY
from langchain_google_genai import ChatGoogleGenerativeAI #Modulo oficial de langchain para gemini
from langchain_core.messages import SystemMessage, HumanMessage 

logger = logging.getLogger(__name__)

def obtener_base64_diapositivas_local(
    nombre_archivo: str,
    num_diapositivas: int,
    ruta_carpeta: str = "docs"
):
    
    ruta_pdf = os.path.join(ruta_carpeta, nombre_archivo)

    #Abriendo el pdf
    doc = fitz.open(ruta_pdf)
    index_pagina = num_diapositivas -1
    pagina = doc[index_pagina]

    #Convirtiendo a pixeles de 150 dpi
    pix = pagina.get_pixmap(matrix = fitz.Matrix(150/72, 150/72), alpha = False)
    image_bytes = pix.tobytes("png")
    return base64.b64encode(image_bytes).decode("utf-8")

def buscar_sintetizar_diapositiva(
    consulta: str = "",
    imagen_base64: Optional[str] = None,
    top_k: int = 3,
)-> Dict[str, Any]:

    try:
        if imagen_base64:
            # 1. Limpiar prefijo si existe coma
            datos_b64 = imagen_base64.split(",")[-1] if "," in imagen_base64 else imagen_base64
            # 2. Decodificar bytes
            bytes_imagen = base64.b64decode(datos_b64)
            # 3. Vectorizar con CLIP Visión
            vector_consulta = generar_embedding_imagen(bytes_imagen)

        elif consulta:
            vector_consulta = generar_embedding(consulta)

        else: 
            return {"error": "Debes enviar un texto o una imagen"}

    except Exception as e:
        logger.error(f"Error procesando la entrada: {e}")
        return {"error": "No se pudo generar el vector"}    

    resultado_rpc = client.rpc(
        "buscar_similares",
        {
            "query": vector_consulta, 
            "top_k": top_k
        }
    ).execute()

    documentos_recuperados = resultado_rpc.data

    if not documentos_recuperados:
        return {"error": "No se encontraron documentos relevantes"}
        
    primer_doc = documentos_recuperados[0]
    meta = primer_doc.get("metadata", {})
    nombre_archivo = meta.get("archivo", "")
    num_diapositiva = meta.get("diapositiva", 1)

    imagen_b64 = obtener_base64_diapositivas_local(nombre_archivo, num_diapositiva)

    try: 
        if GEMINI_API_KEY:
            # 1. Construir el mensaje del usuario con Texto e Imagen
            message = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": f"Consulta del estudiante: {consulta if consulta else 'Explica el contenido de esta diapositiva.'}"
                    },
                    {
                        "type": "image_url",
                        "image_url": f"data:image/png;base64,{imagen_b64}"
                    }
                ]
            )
            # 2. Instanciar Gemini 2.5 Flash
            gemini_model = ChatGoogleGenerativeAI(
                model="gemini-3.5-flash",
                temperature=0.2,
                google_api_key=GEMINI_API_KEY,
                max_tokens= 800
            )
            # 3. Invocar enviando [SystemMessage, HumanMessage]
            respuesta_llm = gemini_model.invoke([SYSTEM_PROMPT_TUTOR, message])
            return {
                "respuesta": respuesta_llm.content,
                "documentos_recuperados": documentos_recuperados
            }
        else:  
            return {
                "respuesta": f"[Modo Búsqueda Vectorial] Se encontraron {len(documentos_recuperados)} diapositivas. Configura GEMINI_API_KEY en .env.",
                "documentos_recuperados": documentos_recuperados
            }
    except Exception as e:
        logger.error(f"Error en síntesis con Gemini: {e}")
        return {"error": f"Error invocando Gemini API: {str(e)}"}

        


if __name__ == "__main__":
    print("Iniciando Prueba del Motor Semántico...\n")
    
    # 1. Escribe aquí cualquier pregunta sobre tus diapositivas:
    pregunta = "¿Que es un hamiltoneano?"
    
    resultado = buscar_sintetizar_diapositiva(pregunta)
    
    print("=== RESPUESTA DE GEMINI ===")
    print(resultado.get("respuesta"))
    
    print("\n=== DIAPOSITIVA RECUPERADA ===")
    if resultado.get("documentos_recuperados"):
        top_1 = resultado["documentos_recuperados"][0]
        print(f"Archivo: {top_1['metadata'].get('archivo')} - Diapositiva: {top_1['metadata'].get('diapositiva')}")
