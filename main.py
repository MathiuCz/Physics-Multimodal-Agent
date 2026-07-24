from app.routers import vision
from app.routers import function_calling
from fastapi import FastAPI

from app.routers import busqueda, documentos, rag, rag_memory, rag_re_ranker

app = FastAPI()

app.include_router(vision.router, prefix="/vision")
app.include_router(function_calling.router, prefix="/function-calling")
app.include_router(busqueda.router, prefix="/busqueda")
app.include_router(documentos.router, prefix="/documentos")
app.include_router(rag.router, prefix="/rag")
app.include_router(rag_memory.router, prefix="/rag-memory")
app.include_router(rag_re_ranker.router, prefix="/rag-re-ranker")
