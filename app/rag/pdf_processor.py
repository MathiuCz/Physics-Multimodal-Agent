import os
import uuid
import logging
from typing import List, Dict, Any
import fitz  

from app.rag.embedding import generar_embedding_imagen
from app.core.supabase_client import client as supabase

logger = logging.getLogger(__name__) # Mapeo de errores
logger.setLevel(logging.INFO)

def procesar_e_inyectar_presentacion_pdf(
    pdf_bytes: bytes, 
    nombre_archivo: str,
    dpi_renderizado: int = 150
) -> Dict[str, Any]: 
    
    try: 
        #Cargamos el PDF en RAM
        documento_pdf = fitz.open(stream=pdf_bytes, filetype="pdf") #
        total_paginas = len(documento_pdf)
        registros_a_insertar = []

        logger.info(f"Procesando '{nombre_archivo}' ({total_paginas} páginas) a {dpi_renderizado} DPI...")

        #Iteramos hoja por hoja (no TextSplitter para no romper fórmulas)
        for num_hoja in range(total_paginas):
            pagina = documento_pdf[num_hoja]
            numero_hoja_real = num_hoja + 1
            
            #Transformamos la hoja entera a una imagen de alta resolución
            factor_escala = dpi_renderizado / 72.0
            matriz = fitz.Matrix(factor_escala, factor_escala)
            pixmap = pagina.get_pixmap(matrix=matriz, alpha=False) # alpha=False quita transparencia
            imagen_bytes = pixmap.tobytes("png")
            
            #Modelo multimodal (CLIP) para vectorizar la imagen
            vector_embedding = generar_embedding_imagen(imagen_bytes)
            
            # Registro para la Base de Datos
            if vector_embedding:
                registro = {
                    "id": str(uuid.uuid4()), # ID seguro
                    "tipo": "imagen",
                    "contenido": f"Diapositiva {numero_hoja_real} del archivo {nombre_archivo} (Física Cuántica)",
                    "metadata": {
                        "archivo": nombre_archivo,
                        "diapositiva": numero_hoja_real
                    },
                    "embedding": vector_embedding,
                }
                registros_a_insertar.append(registro)
        
        #Inserción Masiva 
        if registros_a_insertar:
            supabase.table("documentos").insert(registros_a_insertar).execute()
            logger.info(f"Éxito: Inyectadas {len(registros_a_insertar)} diapositivas de '{nombre_archivo}'.")
            
        return {"estatus": "ok", "paginas_procesadas": len(registros_a_insertar)}
        
    except Exception as e:
        logger.error(f"Error procesando el PDF {nombre_archivo}: {e}")
        return {"estatus": "error", "error": str(e)}

def procesar_directorio_local(ruta_carpeta: str = "docs"):
    """
    Escanea la carpeta local, lee los PDFs en binario y los inyecta en Supabase.
    """
    # Validamos que la carpeta exista en tu computadora
    if not os.path.exists(ruta_carpeta):
        logger.error(f"La carpeta local '{ruta_carpeta}' no existe.")
        return
        
    archivos = os.listdir(ruta_carpeta)
    pdfs = [f for f in archivos if f.lower().endswith('.pdf')]
    
    if not pdfs:
        logger.warning(f"No se encontraron archivos PDF en la carpeta '{ruta_carpeta}'.")
        return
        
    logger.info(f"Se detectaron {len(pdfs)} PDFs en '{ruta_carpeta}'. Iniciando ciclo de inyección...")
    
    for nombre_pdf in pdfs:
        ruta_completa = os.path.join(ruta_carpeta, nombre_pdf)
        
        # Leemos el archivo en binario
        with open(ruta_completa, "rb") as archivo_fisico:
            mis_bytes = archivo_fisico.read()
            
        # Ejecutamos la inyección limpia
        procesar_e_inyectar_presentacion_pdf(
            pdf_bytes=mis_bytes, 
            nombre_archivo=nombre_pdf
        )

if __name__ == "__main__":
    print("Iniciando Motor de Ingesta Local...")
    procesar_directorio_local("docs")
