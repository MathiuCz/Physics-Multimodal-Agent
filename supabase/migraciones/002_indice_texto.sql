-- Migración 002: índice de texto para búsqueda semántica con bge-m3 (1024-d).
-- Ejecutar en el SQL Editor de Supabase. Es idempotente: se puede correr más de una vez.
-- No toca la columna `embedding` (CLIP) ni la RPC `buscar_similares`: ambas siguen sirviendo
-- las consultas con imagen y la línea base de evaluación.

alter table documentos add column if not exists embedding_texto vector(1024);

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
