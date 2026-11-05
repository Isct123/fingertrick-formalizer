from fractions import Fraction
from math import sqrt, isqrt

def frontier(a, k):
    if not (type(a) is int and type(k) is int and (a >= 2) and (1 <= k <= a)):
        raise ValueError('Expected a>=2 and1<=k<=a')
    leaves = a - k
    if leaves == 0:
        return dict(coefficient=float(a + 1), bridge_fraction=1.0, branch='full', rational_coefficient=Fraction(a + 1), rational_bridge=Fraction(1))
    if k <= 3 or (k - 3) ** 2 <= 4 * leaves:
        root = isqrt(leaves)
        exact = root * root == leaves
        return dict(coefficient=(2 + sqrt(leaves)) ** 2, bridge_fraction=2 / (2 + sqrt(leaves)), branch='leaf_limited', rational_coefficient=Fraction((2 + root) ** 2) if exact else None, rational_bridge=Fraction(2, 2 + root) if exact else None)
    return dict(coefficient=(k + 1) * (a - 3) / (k - 3), bridge_fraction=(k - 3) / (a - 3), branch='equalized', rational_coefficient=Fraction((k + 1) * (a - 3), k - 3), rational_bridge=Fraction(k - 3, a - 3))

def lower_coefficient(a, k, precision=10 ** 8):
    f = frontier(a, k)
    if f['rational_coefficient'] is not None:
        return f['rational_coefficient']
    leaves = a - k
    root = Fraction(isqrt(leaves * precision * precision), precision)
    return Fraction(4 + leaves) + 4 * root

def rational_counts(a, k, budget):
    f = frontier(a, k)
    s = f['rational_bridge']
    if s is None:
        return None
    each_bridge = Fraction(budget) * s / (2 * k)
    each_leaf = Fraction(budget) * (1 - s) / (a - k) if k < a else Fraction(0)
    if each_bridge.denominator != 1 or (k < a and each_leaf.denominator != 1):
        return None
    if each_bridge < 1 or (k < a and each_leaf < 1):
        return None
    return [[int(each_bridge), int(each_bridge)] for _ in range(k)] + [[int(each_leaf), 0] for _ in range(a - k)]

def certify_integer_design(a, total, setup):
    if not (type(total) is int and type(setup) is int and (total > 0) and (setup >= 0)):
        raise ValueError('Expected positive integer resource budget and nonnegative setup cost')
    candidates = []
    bounds = []
    for k in range(1, a + 1):
        budget = total - setup * (a + k)
        if budget < a + k:
            continue
        bound = lower_coefficient(a, k) / budget
        counts = rational_counts(a, k, budget)
        bounds.append(dict(k=k, paths=a + k, attempts=budget, lower_bound=bound))
        if counts is not None:
            candidates.append(dict(k=k, paths=a + k, attempts=budget, counts=counts, risk=frontier(a, k)['rational_coefficient'] / budget))
    if not candidates:
        return dict(certified=False, bounds=bounds, candidate=None)
    chosen = min(candidates, key=lambda d: d['risk'])
    certified = all((chosen['risk'] <= b['lower_bound'] for b in bounds))
    return dict(certified=certified, bounds=bounds, candidate=chosen)
