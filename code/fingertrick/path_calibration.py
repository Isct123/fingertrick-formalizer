from fractions import Fraction
from numbers import Integral, Real
import math
import itertools
import heapq

def _integer(value):
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real) and math.isfinite(value) and (value == int(value)) and (abs(value) <= 2 ** 53):
        return int(value)
    raise ValueError('Exact integer features required (floats must be exactly integral and within 2^53)')

def _prepare(query):
    if not query.get('found'):
        raise ValueError('A live accepted query DAG is required')
    nodes = list(query['nodes'])
    source = query['source']
    sink = query['sink']
    d = query['parameters']
    if type(d) is not int or d < 0 or len(set(nodes)) != len(nodes) or (source not in nodes) or (sink not in nodes):
        raise ValueError('Distinct nodes, valid endpoints and feature dimension required')
    position = {v: i for i, v in enumerate(nodes)}
    outgoing = {v: [] for v in nodes}
    incoming = {v: [] for v in nodes}
    features = []
    for i, arc in enumerate(query['arcs']):
        u, v = (arc['source'], arc['target'])
        if u not in position or v not in position or position[u] >= position[v]:
            raise ValueError('Arcs must advance a supplied topological order')
        row = tuple((_integer(x) for x in arc['features']))
        if len(row) != d:
            raise ValueError('Feature dimensions differ')
        features.append(row)
        outgoing[u].append(i)
        incoming[v].append(i)
    parent = {}
    prefix_x = {source: (0,) * d}
    for u in nodes:
        if u not in prefix_x:
            continue
        for i in outgoing[u]:
            v = query['arcs'][i]['target']
            if v not in prefix_x:
                parent[v] = i
                prefix_x[v] = tuple((a + b for a, b in zip(prefix_x[u], features[i])))
    next_arc = {}
    suffix_x = {sink: (0,) * d}
    for u in reversed(nodes):
        if u == sink:
            continue
        choices = [i for i in outgoing[u] if query['arcs'][i]['target'] in suffix_x]
        if choices:
            i = choices[0]
            next_arc[u] = i
            suffix_x[u] = tuple((a + b for a, b in zip(features[i], suffix_x[query['arcs'][i]['target']])))
    if any((v not in prefix_x or v not in suffix_x for v in nodes)):
        raise ValueError('All nodes must lie on a complete source-to-sink path')
    return dict(nodes=nodes, source=source, sink=sink, d=d, outgoing=outgoing, features=features, parent=parent, next_arc=next_arc, prefix_x=prefix_x, suffix_x=suffix_x)

def _suffix_path(query, prepared, node):
    path = []
    while node != prepared['sink']:
        i = prepared['next_arc'][node]
        path.append(i)
        node = query['arcs'][i]['target']
    return tuple(path)

def _witness_path(query, prepared, i):
    arc = query['arcs'][i]
    u = arc['source']
    prefix = []
    while u != prepared['source']:
        j = prepared['parent'][u]
        prefix.append(j)
        u = query['arcs'][j]['source']
    return tuple(reversed(prefix)) + (i,) + _suffix_path(query, prepared, arc['target'])

def path_features(query, path):
    d = query['parameters']
    result = [0] * d
    node = query['source']
    for i in path:
        if type(i) is not int or not 0 <= i < len(query['arcs']):
            raise ValueError('Invalid arc index')
        arc = query['arcs'][i]
        if arc['source'] != node:
            raise ValueError('Discontinuous path')
        for j, value in enumerate(arc['features']):
            result[j] += _integer(value)
        node = arc['target']
    if node != query['sink']:
        raise ValueError('Incomplete path')
    return tuple(result)

def _add_basis(row, basis):
    v = [Fraction(x) for x in row]
    for pivot, b in basis:
        factor = v[pivot]
        if factor:
            v = [a - factor * c for a, c in zip(v, b)]
    pivot = next((j for j, x in enumerate(v) if x), None)
    if pivot is None:
        return False
    divisor = v[pivot]
    v = [x / divisor for x in v]
    updated = []
    for old_pivot, b in basis:
        factor = b[pivot]
        updated.append((old_pivot, [a - factor * c for a, c in zip(b, v)]))
    updated.append((pivot, v))
    basis[:] = sorted(updated, key=lambda item: item[0])
    return True

def affine_path_basis(query, reference=None):
    prepared = _prepare(query)
    reference = tuple(_suffix_path(query, prepared, prepared['source']) if reference is None else reference)
    ref = path_features(query, reference)
    selected = []
    basis = []
    for i, arc in enumerate(query['arcs']):
        x = tuple((a + b + c for a, b, c in zip(prepared['prefix_x'][arc['source']], prepared['features'][i], prepared['suffix_x'][arc['target']])))
        if _add_basis([a - b for a, b in zip(x, ref)], basis):
            selected.append(_witness_path(query, prepared, i))
    return dict(reference_arc_indices=list(reference), basis_arc_indices=[list(p) for p in selected], rank=len(selected), distinct_path_count=len(selected) + 1, pivot_columns=[p for p, _ in basis], witness_candidates=len(query['arcs']) + 1, path_enumeration_used=False)

def _inverse(matrix):
    n = len(matrix)
    a = [[Fraction(x) for x in row] + [Fraction(i == j) for j in range(n)] for i, row in enumerate(matrix)]
    for j in range(n):
        pivot = next((i for i in range(j, n) if a[i][j]), None)
        if pivot is None:
            raise ValueError('Singular projected contrast basis')
        a[j], a[pivot] = (a[pivot], a[j])
        divisor = a[j][j]
        a[j] = [x / divisor for x in a[j]]
        for i in range(n):
            if i != j and a[i][j]:
                factor = a[i][j]
                a[i] = [x - factor * y for x, y in zip(a[i], a[j])]
    return [row[n:] for row in a]

def _extreme(query, prepared, scores, maximize):
    source, sink = (prepared['source'], prepared['sink'])
    distance = {source: Fraction(0)}
    parent = {}
    for u in prepared['nodes']:
        for i in prepared['outgoing'][u]:
            v = query['arcs'][i]['target']
            candidate = distance[u] + scores[i]
            if v not in distance or (candidate > distance[v] if maximize else candidate < distance[v]):
                distance[v] = candidate
                parent[v] = i
    path = []
    v = sink
    while v != source:
        i = parent[v]
        path.append(i)
        v = query['arcs'][i]['source']
    return (distance[sink], tuple(reversed(path)))

def barycentric_calibration(query, coefficient_limit=2, reference=None):
    limit = Fraction(coefficient_limit)
    if limit <= 1:
        raise ValueError('A coefficient limit strictly larger than one is required')
    prepared = _prepare(query)
    result = affine_path_basis(query, reference)
    ref_path = tuple(result['reference_arc_indices'])
    ref = path_features(query, ref_path)
    paths = [tuple(p) for p in result['basis_arc_indices']]
    pivots = result['pivot_columns']
    r = len(paths)
    swaps = []
    calls = 0
    initial_bound = None
    while r:
        rows = [tuple((a - b for a, b in zip(path_features(query, p), ref))) for p in paths]
        inverse = _inverse([[row[j] for j in pivots] for row in rows])
        extrema = []
        worst = (Fraction(0), None, None, None)
        for coordinate in range(r):
            weights = [inverse[j][coordinate] for j in range(r)]
            scores = [sum((Fraction(feature[j]) * w for j, w in zip(pivots, weights)), Fraction(0)) for feature in prepared['features']]
            offset = sum((Fraction(ref[j]) * w for j, w in zip(pivots, weights)), Fraction(0))
            lo, lo_path = _extreme(query, prepared, scores, False)
            hi, hi_path = _extreme(query, prepared, scores, True)
            calls += 2
            lo -= offset
            hi -= offset
            extrema.append((lo, hi))
            for value, path in ((lo, lo_path), (hi, hi_path)):
                if abs(value) > worst[0]:
                    worst = (abs(value), coordinate, path, value)
        if initial_bound is None:
            initial_bound = worst[0]
        if worst[0] <= limit:
            break
        size, coordinate, path, value = worst
        swaps.append(dict(replaced_coordinate=coordinate, determinant_growth=str(size)))
        paths[coordinate] = path
    if not r:
        extrema = []
        initial_bound = Fraction(0)
        inverse = []
    result.update(basis_arc_indices=[list(p) for p in paths], coefficient_limit=str(limit), coordinate_extrema=[dict(lower=str(lo), upper=str(hi)) for lo, hi in extrema], initial_maximum_coefficient=str(initial_bound), final_maximum_coefficient=str(max((max(abs(lo), abs(hi)) for lo, hi in extrema), default=Fraction(0))), projected_basis_inverse=[[str(x) for x in row] for row in inverse], reference_feature_counts=list(ref), basis_feature_contrasts=[list(tuple((a - b for a, b in zip(path_features(query, p), ref)))) for p in paths], swaps=swaps, optimization_oracle_calls=calls, interpretation='Exact model calibration and deterministic noise amplification certificate; no physical timing coverage')
    return result

def uniform_noise_amplification(query, calibration, max_rank=12):
    if type(max_rank) is not int or max_rank < 0:
        raise ValueError('Nonnegative rank limit required')
    r = calibration['rank']
    if r > max_rank:
        return dict(computed=False, reason='Explicit exponential-in-rank limit', rank=r, max_rank=max_rank)
    prepared = _prepare(query)
    pivots = calibration['pivot_columns']
    ref = calibration['reference_feature_counts']
    inverse = [[Fraction(x) for x in row] for row in calibration['projected_basis_inverse']]
    best = Fraction(0)
    witness = tuple(calibration['reference_arc_indices'])
    for signs in itertools.product((-1, 1), repeat=r):
        weights = [sum((inverse[j][i] * signs[i] for i in range(r)), Fraction(0)) for j in range(r)]
        scores = [sum((Fraction(f[j]) * w for j, w in zip(pivots, weights)), Fraction(0)) for f in prepared['features']]
        offset = sum((Fraction(ref[j]) * w for j, w in zip(pivots, weights)), Fraction(0))
        value, path = _extreme(query, prepared, scores, True)
        value -= offset
        if value > best:
            best = value
            witness = path
    return dict(computed=True, amplification=str(best), witness_arc_indices=list(witness), optimization_oracle_calls=2 ** r, rank=r, scope='Tight deterministic error amplification under a correct additive feature model; no noise-coverage claim')

def allocate_independent_repeats(coefficients, standard_deviations, total_attempts):
    c = [Fraction(str(x)) for x in coefficients]
    sigma = [Fraction(str(x)) for x in standard_deviations]
    if len(sigma) != len(c) + 1 or any((x < 0 for x in sigma)) or type(total_attempts) is not int or (total_attempts < len(sigma)):
        raise ValueError('Reference plus basis standard deviations and a sufficient integer budget required')
    a = [-sum(c, Fraction(0))] + c
    v = [x * x * s * s for x, s in zip(a, sigma)]
    counts = [1] * len(v)
    heap = [(-value / 2, i) for i, value in enumerate(v)]
    heapq.heapify(heap)
    for _ in range(total_attempts - len(v)):
        _, i = heapq.heappop(heap)
        counts[i] += 1
        n = counts[i]
        heapq.heappush(heap, (-v[i] / (n * (n + 1)), i))
    variance = sum((value / n for value, n in zip(v, counts)), Fraction(0))
    return dict(reference_then_basis_attempts=counts, variance=str(variance), reference_then_basis_weights=[str(x) for x in a], scope='Optimal integer allocation for supplied independent variances, one fixed target and equal per-attempt cost')

def model_check_relation(query, calibration, target):
    target = tuple(target)
    x = path_features(query, target)
    ref = calibration['reference_feature_counts']
    r = calibration['rank']
    pivots = calibration['pivot_columns']
    inverse = [[Fraction(v) for v in row] for row in calibration['projected_basis_inverse']]
    c = [sum((Fraction(x[pivots[j]] - ref[pivots[j]]) * inverse[j][i] for j in range(r)), Fraction(0)) for i in range(r)]
    weights = [sum(c, Fraction(0)) - 1] + [-v for v in c] + [Fraction(1)]
    common = math.lcm(*(v.denominator for v in weights))
    integers = [int(v * common) for v in weights]
    divisor = math.gcd(*integers)
    integers = [v // divisor for v in integers]
    return dict(reference_basis_target_weights=integers, target_arc_indices=list(target), target_reconstruction_coefficients=[str(v) for v in c], interpretation='Zero expected weighted time if the unrestricted additive model and common offset are correct; no noise significance claim')
