from pathlib import Path
import json, statistics
ROOT = Path(__file__).resolve().parent

def read(name):
    return json.loads((ROOT / name).read_text())

def worst_mse(pred, target):
    return max((statistics.mean(((p[c] - t[c]) ** 2 for p, t in zip(pred, target))) for c in range(len(pred[0]))))
ty = read('results/ty.json')
jb = read('results/jb.json')
old = read('results/regrip.json')['metrics']
output = {'unfamiliar_case_mae_reduction_percent': 100 * (1 - old['huber_regrip']['trial_mae_s'] / old['count_reference']['trial_mae_s'])}
for case in ['T', 'Y']:
    fit = ty['predictions'][case]['12']
    risk = {k: max((statistics.variance((row[c] for row in rows)) for c in range(len(rows[0])))) for k, rows in fit.items()}
    output[case + '_variance_reduction_percent'] = 100 * (1 - risk['pilot_optimal'] / risk['uniform_all'])
risks = {}
for k in ['pilot_optimal', 'uniform_all']:
    risks[k] = statistics.mean((max((statistics.mean((row[c] for row in ty['cases'][case]['12'][k]['posthoc_last_2_target']['episode_cell_squared_errors'])) for c in range(len(ty['cases'][case]['12'][k]['posthoc_last_2_target']['episode_cell_squared_errors'][0])))) for case in ['T', 'Y']))
output['equal_case_ty_accuracy_reduction_percent'] = 100 * (1 - risks['pilot_optimal'] / risks['uniform_all'])
d = jb['unused_tail_diagnostic']
a = worst_mse(d['shared_predictions_s'], d['target_means_s'])
b = worst_mse(d['direct_predictions_s'], d['target_means_s'])
output['jb_shared_model_error_reduction_percent'] = 100 * (1 - a / b)
assert round(output['unfamiliar_case_mae_reduction_percent'], 1) == 10.3
assert round(output['T_variance_reduction_percent'], 1) == 42.6
assert round(output['Y_variance_reduction_percent'], 1) == -4.0
assert round(output['equal_case_ty_accuracy_reduction_percent'], 1) == 1.8
assert round(output['jb_shared_model_error_reduction_percent'], 1) == 10.2
print(json.dumps(output, indent=2))
