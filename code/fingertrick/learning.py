from collections import Counter, defaultdict
import hashlib, json, math
import numpy as np
from scipy.optimize import nnls
from .workbook import HELD_OUT
from .timing_boundary import PROTOCOL

def validate_executions(graph, executions, phase='TRAIN'):
    if graph.spec.get('requires_physical_review'):
        raise ValueError('Candidate coarse graph has unresolved physical assumptions; collection is not authorized by this diagnostic')
    if not executions:
        raise ValueError('No annotated executions')
    ids = set()
    for r in executions:
        identifier = r.get('execution_id', r.get('id'))
        if not identifier or identifier in ids:
            raise ValueError('Unique execution IDs required')
        ids.add(identifier)
        if r.get('phase') != phase or (phase == 'TRAIN' and r.get('case_name') in HELD_OUT):
            raise ValueError('Held-out leakage or phase mismatch')
        if r.get('protocol') != PROTOCOL:
            raise ValueError('Timing protocol mismatch')
        if not r.get('edge_ids'):
            raise ValueError('Empty or unresolved execution')
        graph.validate_path(r['start'], r['edge_ids'], r['algorithm'], r.get('end'))
        if any((graph.states[r['start']][h]['grip'] != 'FRONT' for h in ('left', 'right'))):
            raise ValueError('Timed executions must start with both thumbs FRONT')
    return executions

def validate_records(graph, records, phase='TRAIN'):
    if graph.spec.get('requires_physical_review'):
        raise ValueError('Candidate coarse graph cannot be fitted as a confirmed physical model')
    if not records:
        raise ValueError('No measured records')
    methods = {r.get('measurement_method') for r in records}
    if len(methods) > 1:
        raise ValueError('Mixed or missing measurement methods cannot be pooled')
    ids = set()
    for r in records:
        if r['id'] in ids:
            raise ValueError('Duplicate attempt ID')
        ids.add(r['id'])
        if r['phase'] != phase or (phase == 'TRAIN' and r['case_name'] in HELD_OUT):
            raise ValueError('Held-out leakage or phase mismatch')
        if not r.get('case_name') or not r.get('session'):
            raise ValueError('Case and session required')
        if r['protocol'] != PROTOCOL:
            raise ValueError('Timing protocol mismatch; historical pilot is not silently reclassified')
        if r['result'] not in ('OK', 'Lockup', 'DNF', 'Unknown'):
            raise ValueError('Explicit outcome required')
        t = r.get('time_s')
        if t is not None and (isinstance(t, bool) or not isinstance(t, (int, float)) or (not math.isfinite(t)) or (t <= 0)):
            raise ValueError('Seconds must be positive and finite')
        if r['result'] in ('OK', 'Lockup') and t is None:
            raise ValueError('Completed attempt requires time')
        if not r.get('edge_ids'):
            raise ValueError('Empty execution')
        graph.validate_path(r['start'], r['edge_ids'], r['algorithm'], r.get('end'))
        if any((graph.states[r['start']][h]['grip'] != 'FRONT' for h in ('left', 'right'))):
            raise ValueError('Timed executions must start with both thumbs FRONT')
    return records

def incidence(graph, paths):
    if hasattr(graph, 'feature_matrix'):
        return graph.feature_matrix(paths)
    index = {k: i for i, k in enumerate(graph.parameter_keys)}
    x = np.zeros((len(paths), len(index)), dtype=float)
    for i, path in enumerate(paths):
        for name in path:
            x[i, index[graph.by_id[name].parameter_key]] += 1
    return x

def design_audit(graph, paths):
    x = incidence(graph, paths)
    rank = int(np.linalg.matrix_rank(x)) if x.size else 0
    counts = x.sum(0)
    singular = np.linalg.svd(x, compute_uv=False) if x.size else np.array([])
    return dict(rows=len(paths), parameters=len(graph.parameter_keys), rank=rank, rank_deficit=len(graph.parameter_keys) - rank, singular_values=singular.tolist(), absent=[k for k, v in zip(graph.parameter_keys, counts) if not v], counts=dict(zip(graph.parameter_keys, counts.astype(int).tolist())), condition_number=float(singular[0] / singular[-1]) if rank == len(graph.parameter_keys) and len(singular) and (singular[-1] > 0) else None)

def select_probes(graph, training_paths, candidates):
    x = incidence(graph, training_paths)
    initial = int(np.linalg.matrix_rank(x)) if x.size else 0
    rank = initial
    selected = []
    rejected = {}
    if rank == len(graph.parameter_keys):
        return dict(initial_rank=rank, final_rank=rank, required_rank=rank, selected=[], rejected={}, minimum_proven_within_pool=True)
    for c in sorted(candidates, key=lambda c: (len(c['edge_ids']), c['id'])):
        try:
            if c.get('phase') != 'TRAIN' or c.get('case_name') in HELD_OUT or (not c.get('evidence')):
                raise ValueError('Probe needs TRAIN provenance and evidence')
            graph.validate_path(c['start'], c['edge_ids'], c['algorithm'], c.get('end'))
        except (ValueError, KeyError) as error:
            rejected[c['id']] = str(error)
            continue
        trial = np.vstack((x, incidence(graph, [c['edge_ids']])))
        new = int(np.linalg.matrix_rank(trial))
        if new > rank:
            selected.append(c['id'])
            x = trial
            rank = new
        if rank == len(graph.parameter_keys):
            break
    return dict(initial_rank=initial, final_rank=rank, required_rank=len(graph.parameter_keys), selected=selected, rejected=rejected, minimum_proven_within_pool=rank == len(graph.parameter_keys), scope='Rank completion only; not repetition precision or all-pose coverage')

def clean_records(graph, records):
    validate_records(graph, records)
    clean = [r for r in records if r['result'] == 'OK']
    if not clean:
        raise ValueError('No confirmed OK training timings')
    return clean

def grouped_split(records, validation_fraction=0.2, seed=2026):
    if not 0 < validation_fraction < 1:
        raise ValueError('Invalid validation fraction')
    cases = sorted({r['case_name'] for r in records})
    if len(cases) < 2:
        raise ValueError('At least two distinct TRAIN cases needed for validation')
    rng = np.random.default_rng(seed)
    rng.shuffle(cases)
    n = max(1, min(len(cases) - 1, int(round(len(cases) * validation_fraction))))
    val = set(cases[:n])
    return ([r for r in records if r['case_name'] not in val], [r for r in records if r['case_name'] in val])

def metrics(y, prediction):
    y = np.asarray(y, dtype=float)
    p = np.asarray(prediction, dtype=float)
    if y.shape != p.shape or not len(y) or (not np.isfinite(y).all()) or (not np.isfinite(p).all()):
        raise ValueError('Finite matching observations required')
    e = p - y
    return dict(n=len(y), mae_s=float(np.mean(abs(e))), rmse_s=float(np.sqrt(np.mean(e * e))), bias_s=float(e.mean()), r2=float(1 - np.sum(e * e) / np.sum((y - y.mean()) ** 2)) if np.var(y) > 0 else None)

def fit_regression(graph, records, allow_rank_deficient=False, ridge=0.0):
    clean = clean_records(graph, records)
    if not math.isfinite(ridge) or ridge < 0:
        raise ValueError('Nonnegative ridge required')
    x = incidence(graph, [r['edge_ids'] for r in clean])
    y = np.array([r['time_s'] for r in clean])
    audit = design_audit(graph, [r['edge_ids'] for r in clean])
    if audit['rank_deficit'] and (not allow_rank_deficient):
        raise ValueError('Unidentified edge costs: complete rank or explicitly request a regularized predictive fit')
    if audit['rank_deficit'] and ridge <= 0:
        raise ValueError('Rank-deficient predictive fit requires positive ridge')
    fit_x = np.vstack((x, np.sqrt(ridge) * np.eye(x.shape[1]))) if ridge else x
    fit_y = np.r_[y, np.zeros(x.shape[1])] if ridge else y
    theta, _ = nnls(fit_x, fit_y, maxiter=max(1000, 10 * x.shape[1]))
    ols = np.linalg.lstsq(x, y, rcond=None)[0]
    return dict(model='nonnegative_path_regression', protocol=PROTOCOL, graph_fingerprint=graph.fingerprint, measurement_method=clean[0].get('measurement_method'), weights_s=dict(zip(graph.parameter_keys, theta.tolist())), training_case_names=sorted({r['case_name'] for r in clean}), training_record_hash=hashlib.sha256(json.dumps(clean, sort_keys=True).encode()).hexdigest(), audit=audit, ridge=ridge, individual_costs_identifiable=not audit['rank_deficit'], training_metrics=metrics(y, x @ theta), ols_training_metrics=metrics(y, x @ ols), excluded_outcomes=dict(Counter((r['result'] for r in records if r['result'] != 'OK'))))

def predict(graph, artifact, records):
    if artifact['graph_fingerprint'] != graph.fingerprint or artifact['protocol'] != PROTOCOL:
        raise ValueError('Model graph/protocol mismatch')
    weights = artifact['weights_s']
    if set(weights) != set(graph.parameter_keys):
        raise ValueError('Weight vocabulary mismatch')
    for r in records:
        if artifact.get('measurement_method') and r.get('measurement_method') != artifact['measurement_method']:
            raise ValueError('Prediction measurement method differs from fitted model')
        graph.validate_path(r['start'], r['edge_ids'], r['algorithm'], r.get('end'))
    return incidence(graph, [r['edge_ids'] for r in records]) @ np.array([weights[k] for k in graph.parameter_keys])

def compare_models(graph, records, epochs=300, seed=2026, ridge=0.0001):
    from .gnn import fit_gnn
    from .notation_intake import parse_extended
    clean = clean_records(graph, records)
    train, val = grouped_split(clean, seed=seed)
    linear = fit_regression(graph, train, allow_rank_deficient=True, ridge=ridge)
    linear['validation_case_names'] = sorted({r['case_name'] for r in val})
    linear['validation_metrics'] = metrics([r['time_s'] for r in val], predict(graph, linear, val))
    gnn = fit_gnn(graph, clean, epochs=epochs, seed=seed)
    baselines = {}
    for mode in ('constant', 'logical_move_count', 'quarter_turn_count'):

        def features(rows):
            counts = []
            for r in rows:
                t = [t for t in parse_extended(r['algorithm']) if t.kind != 'ROTATION']
                n = len(t) if mode == 'logical_move_count' else sum((2 if a.turns == 2 else 1 for a in t))
                counts.append([1.0] if mode == 'constant' else [1.0, float(n)])
            return np.array(counts)
        x = features(train)
        y = np.array([r['time_s'] for r in train])
        theta, _ = nnls(x, y, maxiter=1000)
        baselines[mode] = dict(coefficients=theta.tolist(), validation_metrics=metrics([r['time_s'] for r in val], features(val) @ theta))
    families = sorted({(e.kind, e.family) for e in graph.edges})

    def family_x(rows):
        a = np.zeros((len(rows), len(families)))
        for i, r in enumerate(rows):
            for eid in r['edge_ids']:
                e = graph.by_id[eid]
                a[i, families.index((e.kind, e.family))] += 1
        return a
    theta, _ = nnls(family_x(train), np.array([r['time_s'] for r in train]))
    baselines['independent_family_cost'] = dict(columns=families, coefficients=theta.tolist(), validation_metrics=metrics([r['time_s'] for r in val], family_x(val) @ theta))
    return dict(training_cases=sorted({r['case_name'] for r in train}), validation_cases=sorted({r['case_name'] for r in val}), regression=linear, gnn=gnn, baselines=baselines, preferred_by_internal_validation='regression' if linear['validation_metrics']['mae_s'] <= gnn['validation_metrics']['mae_s'] else 'gnn', caveat='Validation is internal to TRAIN; final held-out cases remain untouched. Logical counts are explicitly defined here, not claims of WCA move metrics.')
