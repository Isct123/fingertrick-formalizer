from fractions import Fraction
from math import ceil, floor, isqrt, sqrt
from .support_cost_design import frontier

def feasible_max(a, total, setup):
    if setup == 0:
        return a if total > 0 else 0
    bound = Fraction(total) / Fraction(setup) - a
    return min(a, ceil(bound) - 1)

def risk(a, k, total, setup):
    budget = total - setup * (a + k)
    return frontier(a, k)['coefficient'] / budget if budget > 0 else float('inf')

def five_candidate_optimum(a, total, setup, lower=1, upper=None):
    if type(a) is not int or a < 2 or total <= 0 or (setup < 0):
        raise ValueError('Expected a>=2, positive total and nonnegative setup')
    hi = min(a if upper is None else upper, feasible_max(a, total, setup))
    lo = max(1, lower)
    if lo > hi:
        raise ValueError('No feasible identifying support in requested interval')
    if setup == 0:
        candidates = {hi}
    else:
        threshold = min(a - 1, 1 + isqrt(4 * (a - 2)))
        candidates = set()
        leaf_hi = min(hi, threshold)
        if lo <= leaf_hi:
            candidates.update((lo, leaf_hi))
        left = max(lo, threshold + 1)
        right = min(hi, a - 1)
        if left <= right:
            stationary = 2 * sqrt(total / setup - a + 1) - 1
            clipped = max(left, min(right, stationary))
            candidates.update((max(left, min(right, floor(clipped))), max(left, min(right, ceil(clipped)))))
        if lo <= a <= hi:
            candidates.add(a)
    assert len(candidates) <= 5
    best = min(candidates, key=lambda k: (risk(a, k, total, setup), k))
    return dict(k=best, support=a + best, risk=risk(a, best, total, setup), candidates=sorted(candidates), interval=[lo, hi])

def robust_optimum(a, total, cost_low, cost_high):
    if not 0 <= cost_low <= cost_high:
        raise ValueError('Expected 0<=cost_low<=cost_high')
    common_hi = feasible_max(a, total, cost_high)
    if common_hi < 1:
        raise ValueError('No identifying support feasible at the highest setup cost')
    low = five_candidate_optimum(a, total, cost_low)
    high = five_candidate_optimum(a, total, cost_high)
    qlow, qhigh = (1 / low['risk'], 1 / high['risk'])
    if cost_low == cost_high:
        candidates = set(low['candidates'])
        switch = None
    else:
        switch = total * (qlow - qhigh) / (qlow * cost_high - qhigh * cost_low) - a
        cut = floor(switch)
        candidates = set()
        for c, left, right in [(cost_low, 1, min(common_hi, cut)), (cost_high, max(1, cut + 1), common_hi)]:
            if left <= right:
                candidates.update(five_candidate_optimum(a, total, c, left, right)['candidates'])

    def ratios(k):
        return [risk(a, k, total, cost_low) / low['risk'], risk(a, k, total, cost_high) / high['risk']]
    best = min(candidates, key=lambda k: (max(ratios(k)), k))
    assert len(candidates) <= 10
    return dict(k=best, support=a + best, worst_ratio=max(ratios(best)), endpoint_ratios=ratios(best), candidates=sorted(candidates), endpoint_oracles=[low, high], switch_k=switch, scope='Support chosen first; real-valued repetitions reallocated after setup cost is revealed.')
