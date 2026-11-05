from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import json
import numpy as np
from .composite_graph import load_graph
from .domain_prediction import describe_execution
ROOT = Path(__file__).resolve().parents[1]

def load_inventory(root=ROOT):

    def read(name):
        return json.loads((root / name).read_text(encoding='utf-8'))
    graph = load_graph(read('data/frozen_20260928/selected_graph.json'))
    records = []
    for r in read('outputs/expanded_training_20260930/training_records.json'):
        records.append(dict(r, observation_id='development/' + r['id'], collection=r['source'], actual_execution_id=r.get('execution_id', r['id'])))
    definitions = {v['id']: dict(v, case_name=v['case']) for v in read('outputs/consolidated_final_20260929/experiment_plan.json')['variants']}
    confirmed = read('outputs/confirmed_results_20260930/analysis.json')
    remapped = 0
    for r in confirmed['records']:
        if not isinstance(r['time_s'], (int, float)):
            continue
        actual = r['participant_reported_execution_id']
        v = definitions[actual]
        remapped += actual != r['variant_id']
        records.append(dict(id='historical/' + str(r['run']), observation_id='historical/' + str(r['run']), case_name=v['case_name'], algorithm=v['algorithm'], start=v['start'], edge_ids=v['edge_ids'], time_s=r['time_s'], collection='historical_350', source='historical_350', outcome='CompletedUserConfirmed', actual_execution_id=actual, listed_execution_id=r['variant_id'], session=r['session']))
    for r in read('outputs/new_pll_analysis_20261102/new_trials.json'):
        v = definitions[r['variant']]
        records.append(dict(id='newTY/' + str(r['run']), observation_id='newTY/' + str(r['run']), case_name=v['case_name'], algorithm=v['algorithm'], start=v['start'], edge_ids=v['edge_ids'], time_s=r['time_s'], collection='new_TY_112', source='new_TY_112', outcome='OK', actual_execution_id=v['id']))
    practical = {v['execution_id']: v for v in read('data/compiled/practical_executions.json')}
    for r in read('outputs/more_pll_results_20261102/new_trials.json'):
        v = practical[r['execution_id']]
        records.append(dict(id='newSix/' + str(r['run']), observation_id='newSix/' + str(r['run']), case_name=r['case'], algorithm=v['algorithm'], start=v['start'], edge_ids=v['edge_ids'], time_s=r['time_s'], collection='new_six_PLL_48', source='new_six_PLL_48', outcome='OK', actual_execution_id=v['execution_id']))
    pilot = read('outputs/factorial_pilot_20261102/manifest_CLEAR.json')
    yp = {p['id']: p for p in pilot['paths']}
    ycase = next((c for c in read('outputs/factorial_calibration_20261102/analysis.json')['cases'] if c['case'] == 'PLL Y'))
    for r in read('outputs/factorial_pilot_results_20261102/new_trials.json'):
        v = yp[r['path_id']]
        records.append(dict(id='newY/' + str(r['run']), observation_id='newY/' + str(r['run']), case_name='PLL Y', algorithm=ycase['algorithm'], start=graph.by_id[v['edge_ids'][0]].source, edge_ids=v['edge_ids'], time_s=r['time_s'], collection='new_Y_factorial_48', source='new_Y_factorial_48', outcome='OK', actual_execution_id=v['id'], phase=r['phase']))
    assert len(records) == len({r['observation_id'] for r in records}) == 1190
    assert remapped == 30
    grouped = defaultdict(list)
    for r in records:
        if not np.isfinite(r['time_s']) or r['time_s'] <= 0:
            raise ValueError('Invalid timing')
        graph.validate_path(r['start'], r['edge_ids'], r['algorithm'])
        grouped[r['case_name'], tuple(r['edge_ids'])].append(r)
    priority = {s: i for i, s in enumerate(('original_480', 'earlier_six_cases_60', 'earlier_oll38_92', 'historical_350', 'new_TY_112', 'new_six_PLL_48', 'new_Y_factorial_48'))}
    rows = []
    for (case, edges), rr in sorted(grouped.items()):
        row = dict(rr[0])
        row['path_id'] = hashlib.sha256((case + '|' + ','.join(edges)).encode()).hexdigest()[:16]
        row['trials'] = rr
        values = np.array([r['time_s'] for r in rr])
        row['mean_s'] = float(values.mean())
        row['median_s'] = float(np.median(values))
        original = [r['time_s'] for r in rr if r['collection'] == 'original_480']
        row['original_mean_s'] = float(np.mean(original)) if original else None
        latest = max((r['collection'] for r in rr), key=lambda s: priority[s])
        row['latest_collection'] = latest
        row['latest_mean_s'] = float(np.mean([r['time_s'] for r in rr if r['collection'] == latest]))
        row['description'] = describe_execution(graph, row)
        rows.append(row)
    summary = dict(recorded_archive_total=1218, usable_exact_path_completed_attempts=len(records), unrepresented_control_attempts=20, excluded_reported_failures=8, duplicate_label_attempts_remapped_to_actual_execution=remapped, cases=len({r['case_name'] for r in rows}), physical_paths=len(rows), collection_counts=dict(Counter((r['collection'] for r in records))), scope='All usable observations are retrospective development; original prospective results remain unchanged', latest_collection_priority=priority)
    return (graph, records, rows, summary)
