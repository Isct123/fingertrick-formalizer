import numpy as np

def contrast_design(features, cases, path_ids):
    reference = {}
    rows = []
    for i in path_ids:
        case = cases[i]
        if case in reference:
            rows.append(features[i] - features[reference[case]])
        else:
            reference[case] = i
    return np.array(rows, dtype=float).reshape(-1, features.shape[1])

def certify(design, query, tolerance=1e-09):
    design = np.asarray(design, float)
    query = np.asarray(query, float)
    if len(design):
        _, s, vt = np.linalg.svd(design, full_matrices=False)
        rank = int(sum(s > tolerance * max(1.0, s[0])))
        projection = query @ vt[:rank].T @ vt[:rank]
    else:
        rank = 0
        projection = np.zeros_like(query)
    residual = query - projection
    norm = float(np.linalg.norm(residual))
    relative = norm / max(1.0, float(np.linalg.norm(query)))
    identifiable = relative < tolerance
    result = dict(rank=rank, identifiable=bool(identifiable), relative_unobserved_component=relative)
    if identifiable:
        coefficients = np.linalg.lstsq(design.T, query, rcond=tolerance)[0] if len(design) else np.zeros(0)
        result['reconstruction_coefficients'] = coefficients.tolist()
        result['reconstruction_max_error'] = float(np.max(abs(design.T @ coefficients - query))) if len(query) else 0.0
    else:
        witness = residual / float(query @ residual)
        result['unit_effect_null_witness'] = witness.tolist()
        result['training_contrast_effect_max'] = float(np.max(abs(design @ witness))) if len(design) else 0.0
        result['query_effect'] = float(query @ witness)
    return result
