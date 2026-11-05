from functools import lru_cache
import numpy as np

def matrix(a):
    return np.array([[int(r == i) for r in range(a)] + [j] for i in range(a) for j in (0, 1)], float)

@lru_cache(None)
def allocations(m, budget, cap):

    def generate(left, slots, prefix):
        if slots == 1:
            if 0 <= left <= cap:
                yield (prefix + (left,))
            return
        for n in range(max(0, left - cap * (slots - 1)), min(cap, left) + 1):
            yield from generate(left - n, slots - 1, prefix + (n,))
    return np.array(list(generate(budget, m, ())), int)

def candidates(row_variances, budget, cap=6):
    v = np.asarray(row_variances, float)
    if len(v) < 2 or np.any(~np.isfinite(v)) or np.any(v <= 0):
        raise ValueError('Positive finite row variances required')
    x = matrix(len(v))
    counts = allocations(len(x), budget, cap)
    raw = np.einsum('mi,nm,mj->nij', x, counts, x)
    eligible = np.linalg.matrix_rank(raw) == x.shape[1]
    counts = counts[eligible]
    info = np.einsum('mi,nm,mj->nij', x, counts / np.repeat(v, 2), x)
    risks = np.einsum('mi,nij,mj->nm', x, np.linalg.inv(info), x).max(axis=1)
    return (counts, risks)

def policies(row_variances, budget, episode=0, seed=20261104):
    counts, risks = candidates(row_variances, budget)
    m = counts.shape[1]
    a = m // 2
    best = int(np.argmin(risks))
    full = np.flatnonzero(np.all(counts > 0, axis=1))
    uniform_distance = np.sum((counts[full] - budget / m) ** 2, axis=1)
    uniform_ties = full[np.isclose(uniform_distance, uniform_distance.min(), rtol=0, atol=1e-10)]
    uniform = int(uniform_ties[episode % len(uniform_ties)])
    desired = budget * np.repeat(row_variances, 2) / (2 * np.sum(row_variances))
    distance = np.sum((counts[full] - desired) ** 2, axis=1)
    ties = full[np.isclose(distance, distance.min(), rtol=0, atol=1e-10)]
    proportional = int(ties[np.argmin(risks[ties])])
    supports = sorted(set((tuple(c > 0) for c in counts if np.count_nonzero(c) == a + 1)))
    rng = np.random.default_rng(seed + 1000 * a + 100 * budget + episode)
    support = np.array(supports[int(rng.integers(len(supports)))])
    eligible = np.flatnonzero(np.all((counts > 0) == support, axis=1))
    random = int(eligible[np.argmin(risks[eligible])])
    result = {name: dict(counts=counts[i].tolist(), predicted_worst_variance=float(risks[i])) for name, i in [('pilot_optimal', best), ('uniform_all', uniform), ('variance_proportional', proportional), ('random_support', random)]}
    unbounded, urisk = candidates(row_variances, budget, cap=budget)
    result['integer_optimality_check'] = dict(cap_per_cell=6, capped_minimum=float(risks[best]), uncapped_minimum=float(urisk.min()), cap_binds=bool(risks[best] > urisk.min() * (1 + 1e-09)))
    return result

def predict(row_variances, counts, streams):
    counts = np.asarray(counts, int)
    x = matrix(len(row_variances))
    v = np.repeat(row_variances, 2)
    means = np.array([np.mean(streams[i][:n]) if n else 0.0 for i, n in enumerate(counts)])
    weights = counts / v
    return x @ np.linalg.solve(x.T @ (weights[:, None] * x), x.T @ (weights * means))

def pilot_variances(streams):
    values = np.array([np.var(s, ddof=1) for s in streams])
    return np.maximum(values.reshape(-1, 2).mean(axis=1), 1e-06)
