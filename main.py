from fastapi import FastAPI
from app.routers import busqueda

#Instancia principal
app = FastAPI(
    title="API Motor Semántico RAG Multimodal",
    description="Motor de búsqueda semántica y síntesis RAG para diapositivas con CLIP y Gemini.",
    version="1.0.0"
)

#Endpoint de verificación de estado
@app.get("/", tags=["Health Check"],  description="Verifica el estado del servidor")
def health_check():

    return {
        "status": "online",
        "service": "Motor Semántico RAG Multimodal",
        "version": "1.0.0"
    }

#Registro de routers activos
app.include_router(busqueda.router, prefix="/busqueda", tags=["Búsqueda y RAG"])

