"""Tabu search sobre el campo de radios: un componente a la vez.

El movimiento es cambiar el radio de UN componente. Eso encaja con la
estructura del problema de dos formas. La regresion de la fila i solo depende
de r_i, asi que un movimiento local toca una sola fila de L. Y el objetivo
tiene estructura local, asi que explorar por vecindario en vez de saltar los 40
componentes a la vez conserva lo bueno de la configuracion actual.

La lista tabu prohibe revisitar el par (componente, radio) durante unas
iteraciones, que es lo que evita quedarse dando vueltas en el primer minimo.
"""
import numpy as np

def optimize(obj, lo, hi, budget, rng, x0=None, tenure=8, iters=200,
             n_candidates=None, aspiration=True, **_):
    """Tabu search over the integer radius vector, one cluster at a time."""
    from .base import INF, make_info, out_of_budget, safe_eval
    lo = np.atleast_1d(np.asarray(lo)); hi = np.atleast_1d(np.asarray(hi))
    n = lo.size
    r_min, r_max = int(lo[0]), int(hi[0])
    r = (np.full(n, r_min, dtype=int) if x0 is None
         else np.clip(np.asarray(x0, dtype=float).round().astype(int), r_min, r_max))
    f = safe_eval(obj, r)
    best_r, best_f = r.copy(), f
    tabu = {}
    n_cand = n_candidates or n
    for it in range(int(iters)):
        if out_of_budget(obj):
            break
        comps = rng.permutation(n)[:n_cand]
        cand = (None, INF, None)
        for j in comps:
            for v in range(r_min, r_max + 1):
                if v == r[j] or out_of_budget(obj):
                    continue
                is_tabu = tabu.get((int(j), v), 0) > it
                t = r.copy(); t[j] = v
                ft = safe_eval(obj, t)
                if is_tabu and not (aspiration and ft < best_f):
                    continue
                if ft < cand[1]:
                    cand = ((int(j), v), ft, t)
        if cand[0] is None:
            break
        (j, v), f, r = cand
        tabu[(j, int(r[j]))] = it + int(tenure)
        if f < best_f:
            best_r, best_f = r.copy(), f
    return best_r.astype(float), make_info(obj, best_r, best_f,
                                           dict(optimizer="tabu", tenure=int(tenure)))


def _legacy_tabu(evaluate, n, r_min=1, r_max=12, r0=None, iters=200,
                 tenure=8, n_candidates=None, rng=None, aspiration=True):
    rng = np.random.default_rng() if rng is None else rng
    r = (np.full(n, r_min, dtype=int) if r0 is None
         else np.clip(np.asarray(r0, int), r_min, r_max))
    f = evaluate(r)
    best_r, best_f = r.copy(), f
    tabu = {}
    n_cand = n_candidates or n
    trace = [best_f]
    n_eval = 1

    for it in range(iters):
        comps = rng.permutation(n)[:n_cand]
        cand_best = (None, np.inf, None)
        for j in comps:
            for v in range(r_min, r_max + 1):
                if v == r[j]:
                    continue
                is_tabu = tabu.get((j, v), 0) > it
                t = r.copy(); t[j] = v
                ft = evaluate(t); n_eval += 1
                # aspiracion: un movimiento tabu se acepta si mejora el mejor
                if is_tabu and not (aspiration and ft < best_f):
                    continue
                if ft < cand_best[1]:
                    cand_best = ((j, v), ft, t)
        if cand_best[0] is None:
            break
        (j, v), f, r = cand_best[0], cand_best[1], cand_best[2]
        tabu[(j, int(r[j]))] = it + tenure
        if f < best_f:
            best_r, best_f = r.copy(), f
        trace.append(best_f)
    return best_r, best_f, dict(n_eval=n_eval, trace=trace)
