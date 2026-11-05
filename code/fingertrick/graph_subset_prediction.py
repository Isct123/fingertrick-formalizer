from collections import Counter
import hashlib
import numpy as np
from scipy.optimize import nnls
from scipy.sparse.csgraph import minimum_spanning_tree
from .domain_prediction import fit_predict, _chunk_kernel

def edit_distance(a, b):
    prior = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1] + 1, prior[j] + 1, prior[j - 1] + (x != y)))
        prior = current
    return prior[-1]

def action_word(graph, row):
    moves = [graph.by_id[e] for e in row['edge_ids'] if graph.by_id[e].kind == 'MOVE']
    words = []
    for mirror in (False, True):
        words.append(tuple(((e.family, {'LEFT': 'RIGHT', 'RIGHT': 'LEFT', 'BOTH': 'BOTH'}[e.hand] if mirror else e.hand, e.concurrent) for e in moves)))
    return min(words)

def make_path_topology(graph, rows):
    descriptions = [r['description'] for r in rows]
    cosine = np.clip(_chunk_kernel(descriptions, descriptions), 0, 1)
    words = [action_word(graph, r) for r in rows]
    n = len(rows)
    edit = np.zeros((n, n))
    for i in range(n):
        for j in range(i):
            edit[i, j] = edit[j, i] = edit_distance(words[i], words[j]) / max(1, len(words[i]), len(words[j]))
    distance = 0.5 * (1 - cosine) + 0.5 * edit
    np.fill_diagonal(distance, 0)
    bandwidth = max(float(np.median(distance[distance > 1e-10])), 1e-08)
    adjacency = np.zeros((n, n))
    for i in range(n):
        ids = np.argsort(distance[i], kind='stable')
        ids = ids[ids != i][:8]
        adjacency[i, ids] = np.exp(-distance[i, ids] / bandwidth)
    adjacency = np.maximum(adjacency, adjacency.T)
    degree = adjacency.sum(1)
    laplacian = np.eye(n) - adjacency / np.sqrt(degree[:, None] * degree[None, :])
    eigen, vectors = np.linalg.eigh(laplacian)
    eigen = np.maximum(eigen, 0)
    tree_cost = distance.copy() + 1e-06
    np.fill_diagonal(tree_cost, 0)
    tree = minimum_spanning_tree(tree_cost).toarray()
    tree = tree + tree.T
    conductance = np.zeros_like(tree)
    present = tree > 0
    conductance[present] = 1 / np.maximum(tree[present], 0.02)
    tree_laplacian = np.diag(conductance.sum(1)) - conductance
    inverse = np.linalg.pinv(tree_laplacian, hermitian=True)
    resistance = np.maximum(0, np.diag(inverse)[:, None] + np.diag(inverse)[None, :] - 2 * inverse)
    rs = max(float(np.median(resistance[resistance > 1e-10])), 1e-08)
    kernels = {}
    for tau in (0.3, 3.0, 30.0):
        kernels['heat', tau] = vectors * np.exp(-tau * eigen) @ vectors.T
        kernels['resolvent', tau] = vectors / (1 + tau * eigen) @ vectors.T
        kernels['resistance', tau] = np.exp(-tau * resistance / rs)
    return dict(distance=distance, cosine=cosine, edit=edit, adjacency=adjacency, tree=tree, resistance=resistance, kernels=kernels, scope='Fixed label-free graph on all registered paths; timing labels excluded from graph construction')

def typed_walk_features(graph, row, mirror_tied=True):
    variants = []
    for mirror in (False, True) if mirror_tied else (False,):
        word = []
        order = ('right', 'left') if mirror else ('left', 'right')
        for eid in row['edge_ids']:
            e = graph.by_id[eid]
            hand = {'LEFT': 'RIGHT', 'RIGHT': 'LEFT', 'BOTH': 'BOTH'}[e.hand] if mirror else e.hand
            word.append((e.kind, e.family, hand, tuple((graph.states[e.source][h]['grip'] for h in order)), tuple((graph.states[e.target][h]['grip'] for h in order)), e.concurrent))
        variants.append(tuple(word))
    word = min(variants)
    labels = [hashlib.sha256(repr(t).encode()).hexdigest() for t in word]
    result = Counter()
    for radius in range(3):
        for label in labels:
            result[radius, label] += 1 / np.sqrt(radius + 1)
        labels = [hashlib.sha256(repr((labels[i - 1] if i else 'START', v, labels[i + 1] if i + 1 < len(labels) else 'END')).encode()).hexdigest() for i, v in enumerate(labels)]
    return result

def feature_kernel(counts):
    keys = sorted(set().union(*(set(c) for c in counts)))
    matrix = np.array([[c.get(k, 0) for k in keys] for c in counts])
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    return matrix @ matrix.T

def case_weights(groups):
    counts = Counter(groups)
    return np.array([1 / counts[g] for g in groups])

def kernel_predict(kernel, n, train, test, y, groups, alpha):
    train = np.array(train, dtype=int)
    test = np.array(test, dtype=int)
    w = case_weights(np.asarray(groups)[train])
    a = kernel[np.ix_(train, train)] * n[train, None] * n[None, train]
    cross = kernel[np.ix_(test, train)] * n[test, None] * n[None, train]
    covariance = a + alpha * np.diag(1 / w)
    base = np.c_[np.ones(len(train)), n[train]]
    tbase = np.c_[np.ones(len(test)), n[test]]
    ib = np.linalg.solve(covariance, base)
    iy = np.linalg.solve(covariance, np.asarray(y)[train])
    beta = np.linalg.lstsq(base.T @ ib, base.T @ iy, rcond=None)[0]
    theta = np.linalg.solve(covariance, np.asarray(y)[train] - base @ beta)
    return tbase @ beta + cross @ theta

def diverse_cases(train, groups, distance, budget):
    groups = np.asarray(groups)
    train = np.asarray(train, dtype=int)
    cases = sorted(set(groups[train]))
    n = len(cases)
    if n <= budget:
        return train
    indices = [train[groups[train] == c] for c in cases]
    d = np.array([[np.min(distance[np.ix_(a, b)]) for b in indices] for a in indices])
    selected = [int(np.argmin(d.mean(1)))]
    while len(selected) < budget:
        value = np.min(d[:, selected], axis=1)
        value[selected] = -np.inf
        selected.append(int(np.argmax(value)))
    return train[np.isin(groups[train], [cases[i] for i in selected])]

class Predictor:

    def __init__(self, graph, rows):
        self.rows = rows
        self.descriptions = [r['description'] for r in rows]
        self.groups = np.array([r['case_name'] for r in rows])
        self.topology = make_path_topology(graph, rows)
        self.n = np.array([r['description']['values']['turns'] for r in rows])
        self.y = np.array([r['mean_s'] for r in rows])
        self.targets = dict(mean=self.y, median=np.array([r['median_s'] for r in rows]), original=np.array([r['original_mean_s'] if r['original_mean_s'] is not None else np.nan for r in rows]), latest=np.array([r['latest_mean_s'] for r in rows]))
        self.cache = {}
        self.walk_kernel = feature_kernel([typed_walk_features(graph, r) for r in rows])
        self.absolute_walk_kernel = feature_kernel([typed_walk_features(graph, r, False) for r in rows])
        self.corrected_descriptions = []
        self.hand_descriptions = []
        wrist = {'R', 'Rp', 'RW', 'RpW', 'R2', 'R2W'}
        thumb = {'FT', 'FTp', 'UTM'}
        for r in rows:
            edges = [graph.by_id[e] for e in r['edge_ids']]
            moves = [e for e in edges if e.kind == 'MOVE']
            values = dict(r['description']['values'])
            values['thumb'] = sum((e.family in thumb for e in moves))
            self.corrected_descriptions.append(dict(r['description'], values=values))
            extra = dict(values)
            for hand in ('LEFT', 'RIGHT', 'BOTH'):
                extra['events_' + hand] = sum((e.hand == hand for e in moves))
            for hand in ('LEFT', 'RIGHT'):
                extra['wrist_' + hand] = sum((e.hand == hand and e.family in wrist for e in moves))
                extra['thumb_' + hand] = sum((e.hand == hand and e.family in thumb for e in moves))
                extra['prep_' + hand] = sum((e.hand == hand and e.kind == 'PREP' for e in edges))
                extra['regrip_' + hand] = sum((e.hand == hand and e.kind == 'REGRIP' for e in edges))
            extra['middle_flicks'] = sum((e.family == 'UM' for e in moves))
            self.hand_descriptions.append(dict(r['description'], values=extra))

    def predict(self, config, train, test):
        method = config['method']
        alpha = config['alpha']
        train = np.asarray(train, dtype=int)
        test = np.asarray(test, dtype=int)
        key = (tuple(sorted(config.items())), tuple(train), tuple(test))
        if key in self.cache:
            return self.cache[key].copy()
        target = 'mean'
        fit_train = train
        if method == 'rate_original':
            target = 'original'
            fit_train = train[np.isfinite(self.targets[target][train])]
        elif method == 'rate_latest':
            target = 'latest'
        elif method == 'rate_median_center':
            target = 'median'
        elif method == 'diverse_rate':
            fit_train = diverse_cases(train, self.groups, self.topology['distance'], config['budget'])
        y = self.targets[target]
        if method == 'move_count':
            x = np.c_[np.ones(len(fit_train)), self.n[fit_train]]
            w = np.sqrt(case_weights(self.groups[fit_train]))
            coef = nnls(np.vstack((x * w[:, None], np.array([[0, 0], [0, np.sqrt(alpha)]]))), np.r_[y[fit_train] * w, [0, 0]])[0]
            pred = np.c_[np.ones(len(test)), self.n[test]] @ coef
        elif method in ('typed_walk_kernel', 'absolute_walk_kernel', 'walk_rate_hybrid'):
            kernel = self.absolute_walk_kernel if method == 'absolute_walk_kernel' else self.walk_kernel
            if method == 'walk_rate_hybrid':
                x = np.array([list(r['values'].values()) for r in self.descriptions], float)
                motor = x[:, 2:] / x[:, 1, None]
                scale = np.maximum(np.std(motor[train], axis=0), 1e-08)
                motor /= scale
                dd = np.maximum(0, np.sum(motor * motor, axis=1)[:, None] + np.sum(motor * motor, axis=1)[None, :] - 2 * motor @ motor.T)
                td = dd[np.ix_(train, train)]
                band = max(1.0, float(np.median(td[td > 1e-10]))) if np.any(td > 1e-10) else 1.0
                eta = config['mix']
                kernel = (1 - eta) * np.exp(-dd / band) + eta * kernel
            pred = kernel_predict(kernel, self.n, train, test, y, self.groups, alpha)
        elif method.startswith('graph_'):
            kind = method.removeprefix('graph_')
            pred = kernel_predict(self.topology['kernels'][kind, config['tau']], self.n, train, test, y, self.groups, alpha)
        elif method == 'chunk_kernel':
            pred = kernel_predict(self.topology['cosine'], self.n, train, test, y, self.groups, alpha)
        elif method in ('same_category_rate', 'local_rate'):
            result = []
            for t in test:
                ids = train
                if method == 'same_category_rate':
                    category = self.groups[t].split()[0]
                    ids = train[[self.groups[i].split()[0] == category for i in train]]
                else:
                    cases = sorted(set(self.groups[train]))
                    ranked = sorted(cases, key=lambda c: (float(np.min(self.topology['distance'][t, train[self.groups[train] == c]])), c))
                    ids = train[np.isin(self.groups[train], ranked[:config['neighbors']])]
                if len(ids) < 3:
                    ids = train
                p, _ = fit_predict('rate_residual', [self.descriptions[i] for i in ids], y[ids], self.groups[ids], [self.descriptions[t]], alpha)
                result.append(p[0])
            pred = np.array(result)
        else:
            descriptions = self.hand_descriptions if method == 'hand_aware_rate' else self.corrected_descriptions if method == 'thumb_corrected_rate' else self.descriptions
            pred, _ = fit_predict('rate_residual', [descriptions[i] for i in fit_train], y[fit_train], self.groups[fit_train], [descriptions[i] for i in test], alpha)
        if not np.isfinite(pred).all():
            raise ValueError('Nonfinite prediction')
        self.cache[key] = np.asarray(pred)
        return self.cache[key].copy()

def configurations(extended=False):
    result = []
    for method in ('move_count', 'rate_all', 'rate_original', 'rate_latest', 'rate_median_center', 'same_category_rate', 'chunk_kernel'):
        for alpha in (0.01, 1.0, 100.0):
            result.append(dict(method=method, alpha=alpha))
    for budget in (16, 32, 48):
        for alpha in (1.0, 100.0):
            result.append(dict(method='diverse_rate', budget=budget, alpha=alpha))
    for neighbors in (8, 16, 32):
        for alpha in (1.0, 100.0):
            result.append(dict(method='local_rate', neighbors=neighbors, alpha=alpha))
    for method in ('graph_heat', 'graph_resolvent', 'graph_resistance'):
        for tau in (0.3, 3.0, 30.0):
            for alpha in (0.1, 10.0, 100.0):
                result.append(dict(method=method, tau=tau, alpha=alpha))
    if extended:
        for method in ('typed_walk_kernel', 'absolute_walk_kernel', 'hand_aware_rate', 'thumb_corrected_rate'):
            for alpha in (0.01, 1.0, 100.0):
                result.append(dict(method=method, alpha=alpha))
        for mix in (0.1, 0.3, 0.6):
            for alpha in (1.0, 100.0):
                result.append(dict(method='walk_rate_hybrid', mix=mix, alpha=alpha))
    return result
