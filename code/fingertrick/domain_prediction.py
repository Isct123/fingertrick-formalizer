from collections import Counter
import json
import numpy as np
from .notation_intake import parse_extended

def describe_execution(graph, row):
    tokens = parse_extended(row['algorithm'])
    logical = [t for t in tokens if t.kind != 'ROTATION']
    edges = [graph.by_id[e] for e in row['edge_ids']]
    moves = [e for e in edges if e.kind == 'MOVE']
    n = len(logical)
    if n == 0:
        raise ValueError('Nonempty execution required')
    hands = [e.hand for e in moves]
    run = longest = 0
    previous = None
    for h in hands:
        run = run + 1 if h == previous else 1
        longest = max(longest, run)
        previous = h
    same = sum((a == b and a != 'BOTH' for a, b in zip(hands, hands[1:])))
    switch = sum((a != b and 'BOTH' not in (a, b) for a, b in zip(hands, hands[1:])))

    def grip_change(e):
        return sum((graph.states[e.source][h]['grip'] != graph.states[e.target][h]['grip'] for h in ('left', 'right')))
    pll = float(row.get('case_name', row.get('case', '')).startswith('PLL'))
    values = dict(intercept=1.0, turns=n, pll=pll, pll_turns=pll * n, half_turns=sum((t.turns == 2 for t in logical)), wide=sum((t.kind == 'WIDE' for t in logical)), slice=sum((t.kind == 'SLICE' for t in logical)), rotations=sum((t.kind == 'ROTATION' for t in tokens)), front_back=sum((t.symbol in ('F', 'B', 'Fw', 'Bw') for t in logical)), down=sum((t.symbol in ('D', 'Dw') for t in logical)), move_events=len(moves), prep=sum((e.kind == 'PREP' for e in edges)), regrip=sum((e.kind == 'REGRIP' for e in edges)), concurrent=sum((e.concurrent != 'NONE' for e in moves)), grip_changes=sum((grip_change(e) for e in moves)), same_hand=same, hand_switch=switch, longest_hand_run=longest, wrist=sum((e.family in ('R', 'Rp', 'RW', 'RpW', 'R2', 'R2W') for e in moves)), double_flick=sum((e.family in ('U2', 'D2_PINKY_RING') for e in moves)), thumb=sum((e.family in ('FT', 'FTp', 'UTM', 'UM') for e in moves)), unusual=sum(('COARSE' in e.family or 'SIMUL' in e.family for e in moves)))
    windows = Counter()

    def symbol(e, mirror):
        hand = {'LEFT': 'RIGHT', 'RIGHT': 'LEFT', 'BOTH': 'BOTH'}[e.hand] if mirror else e.hand
        return (e.family, hand, e.concurrent)
    for width in (1, 2, 3, 4):
        for i in range(len(moves) - width + 1):
            variants = [tuple((symbol(e, mirror) for e in moves[i:i + width])) for mirror in (False, True)]
            windows[json.dumps(min(variants), separators=(',', ':'))] += 1
    return dict(values=values, windows=windows)

def weights(groups):
    groups = np.asarray(groups)
    return np.array([1.0 / np.sum(groups == g) for g in groups])

def _matrix(rows, names=None):
    names = list(rows[0]['values']) if names is None else names
    x = np.array([[r['values'][k] for k in names] for r in rows], float)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('Finite matching execution descriptors required')
    return x

def _chunk_kernel(a, b):

    def norm(r):
        return np.sqrt(sum((v * v for v in r['windows'].values())))
    na, nb = ([norm(r) for r in a], [norm(r) for r in b])
    return np.array([[sum((v * y['windows'].get(k, 0) for k, v in x['windows'].items())) / max(1e-12, na[i] * nb[j]) for j, y in enumerate(b)] for i, x in enumerate(a)])

def fit_predict(method, train, y, groups, test, alpha):
    if alpha <= 0 or not np.isfinite(alpha):
        raise ValueError('Positive finite regularization required')
    names = list(train[0]['values'])
    x, tx = (_matrix(train, names), _matrix(test, names))
    y = np.asarray(y, float)
    if not len(x) or y.shape != (len(x),) or (not np.isfinite(y).all()):
        raise ValueError('Matching finite training observations required')
    w = weights(groups)
    base, tbase = (x[:, :2], tx[:, :2])
    if method == 'turn_baseline':
        beta = np.linalg.lstsq(base * np.sqrt(w[:, None]), y * np.sqrt(w), rcond=None)[0]
        return (tbase @ beta, dict(baseline=beta.tolist(), feature_names=names))
    if method == 'motor_ridge':
        scale = np.maximum(np.std(x, axis=0), 1.0)
        a, b = (x / scale, tx / scale)
        penalty = np.eye(x.shape[1]) * alpha
        penalty[:2, :2] = 0.0
        beta = np.linalg.lstsq(np.vstack((a * np.sqrt(w[:, None]), np.sqrt(penalty))), np.r_[y * np.sqrt(w), np.zeros(x.shape[1])], rcond=None)[0]
        return (b @ beta, dict(scale=scale.tolist(), coefficients=beta.tolist(), feature_names=names))
    if method == 'chunk_residual':
        kernel, cross = (_chunk_kernel(train, train), _chunk_kernel(test, train))
    elif method in ('motor_residual', 'rate_residual'):
        a, b = (x[:, 2:].copy(), tx[:, 2:].copy())
        if method == 'rate_residual':
            a /= x[:, 1, None]
            b /= tx[:, 1, None]
        scale = np.maximum(np.std(a, axis=0), 1e-08)
        a, b = (a / scale, b / scale)
        distance = np.maximum(0.0, np.sum(a * a, axis=1)[:, None] + np.sum(a * a, axis=1)[None, :] - 2 * a @ a.T)
        bandwidth = max(1.0, float(np.median(distance[distance > 1e-10]))) if np.any(distance > 1e-10) else 1.0
        kernel = np.exp(-distance / bandwidth)
        cross_d = np.maximum(0.0, np.sum(b * b, axis=1)[:, None] + np.sum(a * a, axis=1)[None, :] - 2 * b @ a.T)
        cross = np.exp(-cross_d / bandwidth)
    else:
        raise ValueError('Unknown domain predictor')
    if method == 'rate_residual':
        kernel *= x[:, 1, None] * x[None, :, 1]
        cross *= tx[:, 1, None] * x[None, :, 1]
    covariance = kernel + alpha * np.diag(1 / w)
    inv_base = np.linalg.solve(covariance, base)
    inv_y = np.linalg.solve(covariance, y)
    beta = np.linalg.lstsq(base.T @ inv_base, base.T @ inv_y, rcond=None)[0]
    coefficients = np.linalg.solve(covariance, y - base @ beta)
    artifact = dict(baseline=beta.tolist(), coefficients=coefficients.tolist(), alpha=alpha, feature_names=names, training_turns=x[:, 1].tolist())
    if method != 'chunk_residual':
        artifact.update(scale=scale.tolist(), training_standardized=a.tolist(), bandwidth=bandwidth)
    return (tbase @ beta + cross @ coefficients, artifact)

def predict_artifact(method, artifact, test, training=None):
    x = _matrix(test, artifact['feature_names'])
    base = x[:, :2]
    if method == 'turn_baseline':
        return base @ np.asarray(artifact['baseline'])
    if method == 'motor_ridge':
        return x / np.asarray(artifact['scale']) @ np.asarray(artifact['coefficients'])
    if method == 'chunk_residual':
        if training is None:
            raise ValueError('Training chunk descriptors required')
        cross = _chunk_kernel(test, training)
    elif method in ('motor_residual', 'rate_residual'):
        b = x[:, 2:].copy()
        if method == 'rate_residual':
            b /= x[:, 1, None]
        b /= np.asarray(artifact['scale'])
        a = np.asarray(artifact['training_standardized'])
        distance = np.maximum(0.0, np.sum(b * b, axis=1)[:, None] + np.sum(a * a, axis=1)[None, :] - 2 * b @ a.T)
        cross = np.exp(-distance / artifact['bandwidth'])
        if method == 'rate_residual':
            cross *= x[:, 1, None] * np.asarray(artifact['training_turns'])[None, :]
    else:
        raise ValueError('Unknown saved domain predictor')
    prediction = base @ np.asarray(artifact['baseline']) + cross @ np.asarray(artifact['coefficients'])
    if not np.isfinite(prediction).all():
        raise ValueError('Nonfinite domain prediction')
    return prediction
