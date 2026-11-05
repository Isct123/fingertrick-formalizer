import numpy as np
from scipy.optimize import brentq

def coefficient(a, k, total_variance, double_sum, hub_variance):
    m = float(hub_variance)
    q = max(0.0, float(double_sum) - k * m)
    leaves = float(total_variance) - float(double_sum)
    if k == a:
        z = max((k + 1) / 4, (1 + np.sqrt(q / m)) / 4)
    else:
        lower = max(1.0, (k + 1) / 4)

        def derivative(z):
            return 4 * m - 4 * q / (4 * z - 1) ** 2 - leaves / (z - 1) ** 2
        if lower > 1 and derivative(lower) >= 0:
            z = lower
        else:
            lo = lower if lower > 1 else np.nextafter(1.0, 2.0)
            hi = max(2.0, 2 * lower)
            while derivative(hi) < 0:
                hi *= 2
            z = brentq(derivative, lo, hi, xtol=1e-12, rtol=1e-13)
    aa = 4 * z / (4 * z - 1)
    bb = z / (z - 1) if k < a else 0.0
    value = 4 * m * z + q * aa + leaves * bb
    return dict(coefficient=float(value), z=float(z), double_multiplier=float(aa), single_multiplier=float(bb))

def _candidates(sorted_v, k, prefix):
    a = len(sorted_v)
    total = float(prefix[a])
    tail = float(prefix[a] - prefix[a - k + 1]) if k > 1 else 0.0
    result = []
    for pattern, m, sd in [('noisiest', sorted_v[a - k], float(prefix[a] - prefix[a - k])), ('quiet_hub', sorted_v[0], float(sorted_v[0] + tail))]:
        if k == a and result:
            continue
        result.append(dict(pattern=pattern, k=k, **coefficient(a, k, total, sd, m)))
    return result

def plan(variances, total, setup=0.0, k=None):
    v = np.asarray(variances, float)
    a = len(v)
    if v.ndim != 1 or a < 2 or np.any(~np.isfinite(v)) or np.any(v <= 0) or (total <= 0) or (setup < 0):
        raise ValueError('Expected at least two positive finite row variances and a valid budget')
    order = np.argsort(v, kind='stable')
    sv = v[order]
    prefix = np.r_[0.0, np.cumsum(sv)]
    choices = []
    for size in range(1, a + 1) if k is None else (k,):
        if not 1 <= size <= a:
            raise ValueError('Invalid number of double rows')
        budget = total - setup * (a + size)
        if budget <= 0:
            continue
        for candidate in _candidates(sv, size, prefix):
            choices.append(dict(candidate, risk=candidate['coefficient'] / budget, budget=float(budget)))
    if not choices:
        raise ValueError('No identifying support is affordable')
    best = min(choices, key=lambda x: (x['risk'], x['k'], x['pattern']))
    k = best['k']
    indices = list(order[a - k:]) if best['pattern'] == 'noisiest' else [int(order[0])] + list(order[a - k + 1:]) if k > 1 else [int(order[0])]
    indices = list(map(int, indices))
    hub = min(indices, key=lambda i: v[i])
    selected = set(indices)
    r = best['risk']
    z = best['z']
    aa = best['double_multiplier']
    bb = best['single_multiplier']
    counts = np.zeros((a, 2))
    for i in range(a):
        if i in selected:
            counts[i, :] = v[i] * aa / (2 * r)
        else:
            counts[i, 0] = v[i] * bb / r
    extra = v[hub] * max(0.0, 4 * z - k * aa) / r
    counts[hub, :] += extra / 2
    return dict(best, support=a + k, double_rows=indices, precision_hub=int(hub), counts=counts.tolist(), candidate_count=len(choices))

def robust_plan(variances, total, cost_low, cost_high):
    v = np.asarray(variances, float)
    a = len(v)
    if not 0 <= cost_low <= cost_high:
        raise ValueError('Expected a nonnegative ordered cost interval')
    plan(v, total, cost_high, k=1)
    sv = np.sort(v)
    prefix = np.r_[0.0, np.cumsum(sv)]
    curve = {k: min(_candidates(sv, k, prefix), key=lambda x: x['coefficient']) for k in range(1, a + 1)}

    def value(k, c):
        b = total - c * (a + k)
        return curve[k]['coefficient'] / b if b > 0 else float('inf')
    oracle = []
    for c in (cost_low, cost_high):
        best = min(range(1, a + 1), key=lambda k: value(k, c))
        oracle.append(dict(k=best, support=a + best, risk=value(best, c)))
    allowed = [k for k in range(1, a + 1) if total > cost_high * (a + k)]

    def ratios(k):
        return [value(k, c) / o['risk'] for c, o in zip((cost_low, cost_high), oracle)]
    selected = min(allowed, key=lambda k: (max(ratios(k)), k))
    return dict(k=selected, support=a + selected, pattern=curve[selected]['pattern'], worst_ratio=max(ratios(selected)), endpoint_ratios=ratios(selected), endpoint_oracles=oracle, endpoint_plans=[plan(v, total, c, k=selected) for c in (cost_low, cost_high)], scope='Known row variances; support chosen first, real repetitions allocated after setup cost is revealed.')
