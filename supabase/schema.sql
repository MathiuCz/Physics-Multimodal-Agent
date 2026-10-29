-- Esquema completo del tutor (pgvector). Recrea la base desde cero.
-- La tabla está reconstruida a partir de las RPC y del código de ingesta: verificar contra la base
-- real con information_schema si hay dudas. Los cambios incrementales viven en migraciones/.

create extension if not exists vector;

create table if not exists documentos (
    id               uuid primary key,              -- uuid5(archivo#diapositiva), generado en la ingesta
    tipo             text not null,                 -- 'imagen' para diapositivas
    contenido        text,                          -- documento de búsqueda (texto + descripción)
    metadata         jsonb not null default '{}',   -- {archivo, diapositiva, hash_archivo, version_indice}
    embedding        vector(512),                   -- CLIP ViT-B/32 de la imagen
    embedding_texto  vector(1024)                   -- bge-m3 del documento de búsqueda
);

-- Índices ANN: solo compensan con decenas de miles de filas; con cientos, la búsqueda exacta es mejor.
-- create index on documentos using hnsw (embedding vector_cosine_ops);
-- create index on documentos using hnsw (embedding_texto vector_cosine_ops);

-- CLIP: consultas con imagen (y línea base de texto CLIP)
create or replace function buscar_similares(
    query vector(512),
    top_k int default 3
)
returns table (id uuid, tipo text, contenido text, metadata jsonb, similitud float)
language plpgsql
as $$
begin
    return query
    select d.id, d.tipo, d.contenido, d.metadata,
           1 - (d.embedding <=> query) as similitud
    from documentos d
    order by d.embedding <=> query asc
    limit top_k;
end;
$$;

-- bge-m3: búsqueda semántica sobre el documento de búsqueda (migración 002)
create or replace function buscar_similares_texto(
    query vector(1024),
    top_k int default 3
)
returns table (id uuid, tipo text, contenido text, metadata jsonb, similitud float)
language sql stable
as $$
    select d.id, d.tipo, d.contenido, d.metadata,
           1 - (d.embedding_texto <=> query) as similitud
    from documentos d
    where d.tipo = 'imagen' and d.embedding_texto is not null
    order by d.embedding_texto <=> query
    limit top_k;
$$;
