from pathlib import Path
import json, numpy as np
root = Path(__file__).resolve().parent
rows = json.loads((root / 'data/frozen_cv_row_order.json').read_text())
pred = np.load(root / 'data/frozen_cv_predictions.npz')
cases = sorted({r['case'] for r in rows})
scores = {}
for method, forecasts in pred.items():
    errors = []
    for forecast in forecasts:
        by_case = {c: [] for c in cases}
        for row, yhat in zip(rows, forecast):
            by_case[row['case']].extend(abs(np.asarray(row['times_s']) - yhat))
        errors.append([np.mean(by_case[c]) for c in cases])
    scores[method] = np.mean(errors, axis=0)
draw = np.random.default_rng(20261103).integers(0, len(cases), (10000, len(cases)))
a = scores['huber_regrip']
b = scores['count_reference']
result = dict(reduction_percent=float(100 * (1 - a.mean() / b.mean())), conditional_ci95_percent=np.quantile(100 * (1 - a[draw].mean(1) / b[draw].mean(1)), [0.025, 0.975]).tolist())
assert round(result['reduction_percent'], 1) == 10.3
print(json.dumps(result, indent=2))
