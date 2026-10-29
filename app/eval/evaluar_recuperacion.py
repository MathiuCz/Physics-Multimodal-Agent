"""
Evalúa SOLO el recuperador, sin LLM: hit@1, hit@k y MRR, comparando modos de recuperación.

Lectura diagnóstica:
- hit@1 bajo y hit@k alto → la diapositiva correcta se recupera pero no queda primera: agregar reranker.
- hit@k bajo              → la representación no entiende las diapositivas: cambiarla.

Uso (desde la raíz):
    uv run python -m app.eval.evaluar_recuperacion --k 3                      # compara todos los modos
    uv run python -m app.eval.evaluar_recuperacion --k 3 --modos clip hibrido
    uv run python -m app.eval.evaluar_recuperacion --origen manual            # solo preguntas escritas a mano
"""

import argparse
import json
import logging
from functools import partial
from typing import Any, Callable

from app.eval.generar_dataset import RUTA_SALIDA
from app.rag.retrieval import MODOS, recuperar

Recuperador = Callable[..., list[dict[str, Any]]]


def evaluar(golden: list[dict], recuperador: Recuperador, k: int = 3) -> dict[str, Any]:
    """Calcula hit@1, hit@k y MRR@k. Devuelve también los casos fallidos para inspeccionarlos."""
    hit1 = hitk = rr = 0.0
    fallos: list[dict[str, Any]] = []

    for caso in golden:
        relevantes = {(archivo, int(n)) for archivo, n in caso["relevantes"]}
        ranking = [
            (d["metadata"]["archivo"], int(d["metadata"]["diapositiva"]))
            for d in recuperador(caso["pregunta"], top_k=k)
        ]
        posiciones = [i for i, r in enumerate(ranking, start=1) if r in relevantes]

        hit1 += bool(posiciones and posiciones[0] == 1)
        hitk += bool(posiciones)
        rr += 1 / posiciones[0] if posiciones else 0.0
        if not posiciones or posiciones[0] != 1:
            fallos.append({"pregunta": caso["pregunta"], "esperado": sorted(relevantes), "obtenido": ranking})

    n = len(golden) or 1
    return {"n": len(golden), "hit@1": hit1 / n, f"hit@{k}": hitk / n, f"MRR@{k}": rr / n, "fallos": fallos}


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=RUTA_SALIDA)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--modos", nargs="+", default=list(MODOS), choices=MODOS)
    parser.add_argument("--origen", choices=["sintetico", "manual"], help="filtra las preguntas por origen")
    parser.add_argument("--mostrar-fallos", type=int, default=0, help="fallos a imprimir por modo")
    args = parser.parse_args()

    with open(args.dataset, encoding="utf-8") as f:
        golden = [c for c in json.load(f) if "pregunta" in c]
    if args.origen:
        golden = [c for c in golden if c.get("origen") == args.origen]

    print(f"Casos evaluados: {len(golden)} | k = {args.k}\n")
    print(f"{'modo':<10}{'hit@1':>8}{f'hit@{args.k}':>8}{f'MRR@{args.k}':>8}")
    reportes = {}
    for modo in args.modos:
        reporte = evaluar(golden, partial(recuperar, modo=modo), k=args.k)
        reportes[modo] = reporte
        print(f"{modo:<10}{reporte['hit@1']:>8.3f}{reporte[f'hit@{args.k}']:>8.3f}{reporte[f'MRR@{args.k}']:>8.3f}")

    if args.mostrar_fallos:
        for modo, reporte in reportes.items():
            print(f"\n[{modo}] primeros {args.mostrar_fallos} casos donde la correcta no quedó primera:")
            for fallo in reporte["fallos"][: args.mostrar_fallos]:
                print(f"- {fallo['pregunta']}\n    esperado: {fallo['esperado']}\n    obtenido: {fallo['obtenido']}")
