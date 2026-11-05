import numpy as np
from scipy.linalg import qr
from .parser import FAMILIES, parse_families
COLUMN_NAMES = tuple([f'start:{f}' for f in FAMILIES] + [f'pair:{a}>{b}' for a in FAMILIES for b in FAMILIES])
COLUMN_INDEX = {name: i for i, name in enumerate(COLUMN_NAMES)}

def feature_row(sequence):
    sequence = parse_families(sequence) if isinstance(sequence, str) else tuple(sequence)
    if not sequence:
        raise ValueError('An execution must have at least one physical family')
    if any((f not in FAMILIES for f in sequence)):
        raise ValueError('Family outside the legacy 132-column model; FP/R2 require an explicit revised feature schema')
    row = np.zeros(len(COLUMN_NAMES), dtype=np.int64)
    row[COLUMN_INDEX[f'start:{sequence[0]}']] = 1
    for a, b in zip(sequence, sequence[1:]):
        row[COLUMN_INDEX[f'pair:{a}>{b}']] += 1
    return row

def matrix(sequences):
    return np.array([feature_row(s) for s in sequences], dtype=np.int64).reshape(-1, len(COLUMN_NAMES))

def rank(x):
    return int(np.linalg.matrix_rank(x)) if x.size else 0

def audit(sequences, low_frequency=5):
    if low_frequency < 1:
        raise ValueError('Low-frequency threshold must be positive')
    sequences = [parse_families(s) if isinstance(s, str) else tuple(s) for s in sequences]
    x = matrix(sequences)
    r = rank(x)
    singular = np.linalg.svd(x, compute_uv=False) if len(x) else np.array([])
    counts = x.sum(axis=0)
    active = np.flatnonzero(counts)
    collinear = []
    for offset, i in enumerate(active):
        for j in active[offset + 1:]:
            first = np.flatnonzero(x[:, i])[0]
            if x[first, j] and np.array_equal(x[:, i] * x[first, j], x[:, j] * x[first, i]):
                collinear.append([COLUMN_NAMES[i], COLUMN_NAMES[j]])
    independent = []
    if len(x):
        _, _, pivots = qr(x.astype(float), pivoting=True, mode='economic')
        independent = [COLUMN_NAMES[i] for i in pivots[:r]]
    return dict(rows=len(x), unique_rows=len(np.unique(x, axis=0)), parameters=132, rank=r, rank_deficit=132 - r, minimum_additional_independent_sequences=132 - r, singular_values=singular.tolist(), condition_number=float(singular[0] / singular[-1]) if r == 132 else None, identifiable_condition_number=float(singular[0] / singular[r - 1]) if r else None, rank_tolerance=float(max(x.shape) * np.finfo(float).eps * singular[0]) if len(singular) else 0, family_counts={f: sum((s.count(f) for s in sequences)) for f in FAMILIES}, start_counts={f: int(counts[i]) for i, f in enumerate(FAMILIES)}, pair_counts={name[5:]: int(counts[i]) for i, name in enumerate(COLUMN_NAMES) if i >= 11}, absent_columns=[COLUMN_NAMES[i] for i in range(132) if counts[i] == 0], absent_transitions=[COLUMN_NAMES[i][5:] for i in range(11, 132) if counts[i] == 0], low_frequency_columns={COLUMN_NAMES[i]: int(counts[i]) for i in range(132) if 0 < counts[i] < low_frequency}, low_frequency_threshold=low_frequency, perfectly_collinear_nonzero_columns=collinear, independent_columns=independent, dependent_columns=[c for c in COLUMN_NAMES if c not in independent], interpretation='Counts are per distinct supplied execution row, not multiplied by planned repeats. Null condition_number means the full 132-parameter model is not identifiable.')
