from collections import Counter, deque
from pathlib import Path
import json, csv
import numpy as np
from scipy.linalg import qr
from .execution_graph import ExecutionGraph
from .cube import permutation
from .learning import design_audit

def composite_key(previous, current, granularity='grip_context'):

    def action(row, mirror=False):
        hand = row['hand']
        if mirror:
            hand = {'LEFT': 'RIGHT', 'RIGHT': 'LEFT', 'BOTH': 'BOTH'}[hand]
        order = ('RIGHT', 'LEFT') if mirror else ('LEFT', 'RIGHT')
        return [row['family'], hand, [row['grips_before'][h] for h in order], [row['grips_after'][h] for h in order], row['concurrent']]
    values = []
    for mirror in (False, True):
        a = None if previous is None else action(previous, mirror)
        b = action(current, mirror)
        if granularity == 'canonical':
            if a is not None:
                a = a[:2]
                b = b[:2]
            else:
                b = b[:3]
        values.append(json.dumps(['START' if previous is None else 'PAIR', a, b], separators=(',', ':')))
    return min(values)

def feature_counts(moves, granularity='grip_context'):
    out = Counter()
    previous = None
    for move in moves:
        out[composite_key(previous, move, granularity)] += 1
        previous = move
    return out

def audit_design(graph, executions):
    if not executions or any((e['phase'] != 'TRAIN' for e in executions)):
        raise ValueError('TRAIN executions required')
    vectors = [feature_counts(e['move_info']) for e in executions]
    keys = sorted(set().union(*(set(v) for v in vectors)))

    def row(v):
        return np.array([v.get(k, 0) for k in keys], dtype=float)
    x = np.array([row(v) for v in vectors])
    rank = int(np.linalg.matrix_rank(x))
    q, _, _ = qr(x.T, mode='economic', pivoting=True)
    basis = q[:, :rank]
    candidates = []
    for execution in executions:
        for count in range(1, len(execution['move_info'])):
            moves = execution['move_info'][:count]
            last = moves[-1]
            turns = ' '.join((m['turns'] for m in moves))
            ids = last['path_prefix']
            end = graph.by_id[ids[-1]].target
            graph.validate_path(execution['start'], ids, turns, end)
            candidates.append(dict(id=execution['case_name'].replace(' ', '_') + f'_prefix_{count:02}', case_name=execution['case_name'], phase='TRAIN', execution='Probe', protocol=execution['protocol'], algorithm=turns, start=execution['start'], end=end, edge_ids=ids, move_info=moves, move_blocks=count, evidence='Prefix of user-reported TRAIN execution; coarse candidate path', physical_review_required=True))
    candidates.sort(key=lambda c: (c['move_blocks'], len(c['edge_ids']), c['id']))
    chosen = []
    for candidate in candidates:
        v = row(feature_counts(candidate['move_info']))
        residual = v - basis @ (basis.T @ v)
        residual -= basis @ (basis.T @ residual)
        norm = np.linalg.norm(residual)
        if norm > 1e-09:
            chosen.append(candidate)
            basis = np.column_stack((basis, residual / norm))
        if basis.shape[1] == len(keys):
            break
    combined = np.vstack([x] + [row(feature_counts(c['move_info']))[None, :] for c in chosen])
    final_rank = int(np.linalg.matrix_rank(combined))
    counts = x.sum(0).astype(int)
    report = dict(status='CONDITIONAL_ON_COARSE_GRAPH_REVIEW', training_cases=len(executions), cost_model='Composite start/ordered-pair costs with both grips, handedness, concurrency and whole-reflection tying', target_scope='Only contexts observed in TRAIN; not all possible physical transitions', parameters=len(keys), initial_rank=rank, rank_deficit=len(keys) - rank, rank_with_eight_repetitions=int(np.linalg.matrix_rank(np.repeat(x, 8, axis=0))), prefix_candidates=len(candidates), selected_probes=len(chosen), final_rank=final_rank, minimum_proven_within_prefix_pool=final_rank == len(keys) and len(chosen) == len(keys) - rank, minimum_scope='Each additional distinct sequence adds at most one rank; greedy independent prefix extension reaches this bound. No claim over unknown physical transitions.', column_counts=dict(zip(keys, counts.tolist())), low_frequency=[k for k, n in zip(keys, counts) if n < 5], parameter_keys=keys, all_transition_coverage=False, ready_for_mass_timing=False, independent_edge_cost_audit=design_audit(graph, [e['edge_ids'] for e in executions]))
    return (report, chosen)

def reachable_design(graph, executions):

    def info(e):
        return dict(edge_id=e.id, family=e.family, hand=e.hand, concurrent=e.concurrent, turns=e.physical_turns, grips_before={h.upper(): graph.states[e.source][h]['grip'] for h in ('left', 'right')}, grips_after={h.upper(): graph.states[e.target][h]['grip'] for h in ('left', 'right')})
    moves = {e.id: info(e) for e in graph.edges if e.kind == 'MOVE'}
    start = executions[0]['start']
    if any((e['start'] != start for e in executions)):
        raise ValueError('Common timed starting state required')
    first = (start, None)
    queue = deque([first])
    parent = {first: None}
    first_context = {}

    def path_to(node, edge):
        ids = [edge.id]
        while parent[node] is not None:
            node, prior = parent[node]
            ids.append(prior)
        return ids[::-1]
    while queue:
        node = queue.popleft()
        state, last = node
        for edge in graph.outgoing[state]:
            nxtlast = last
            if edge.kind == 'MOVE':
                key = composite_key(moves.get(last), moves[edge.id], 'canonical')
                if key not in first_context:
                    first_context[key] = path_to(node, edge)
                nxtlast = edge.id
            nxt = (edge.target, nxtlast)
            if nxt not in parent:
                parent[nxt] = (node, edge.id)
                queue.append(nxt)
    keys = sorted(first_context)
    index = {k: i for i, k in enumerate(keys)}

    def sparse(ids):
        counts = feature_counts([moves[i] for i in ids if i in moves], 'canonical')
        return {index[k]: float(v) for k, v in counts.items()}
    basis = {}

    def add(v):
        v = dict(v)
        while v:
            pivot = min(v)
            value = v[pivot]
            if pivot not in basis:
                basis[pivot] = {k: x / value for k, x in v.items()}
                return True
            for k, x in basis[pivot].items():
                new = v.get(k, 0.0) - value * x
                if abs(new) < 1e-09:
                    v.pop(k, None)
                else:
                    v[k] = new
        return False
    rows = [sparse(e['edge_ids']) for e in executions]
    for row in rows:
        add(row)
    initial = len(basis)
    selected = []
    for key, ids in sorted(first_context.items(), key=lambda pair: (sum((i in moves for i in pair[1])), len(pair[1]), pair[0])):
        row = sparse(ids)
        if add(row):
            turns = ' '.join((graph.by_id[i].physical_turns for i in ids if i in moves))
            end = graph.by_id[ids[-1]].target
            graph.validate_path(start, ids, turns, end)
            selected.append(dict(id=f'ALL_{len(selected) + 1:04}', algorithm=turns, start=start, end=end, edge_ids=ids, phase='TRAIN', execution='Probe', protocol=executions[0]['protocol'], physical_review_required=True, evidence='Generated walk in candidate coarse graph; not user-confirmed execution'))
            rows.append(row)
    x = np.array([[row.get(i, 0.0) for i in range(len(keys))] for row in rows])
    verified = int(np.linalg.matrix_rank(x))
    initial_verified = int(np.linalg.matrix_rank(x[:len(executions)]))
    if initial != initial_verified or verified != len(basis):
        raise ValueError('Rank verification disagreement')
    return (dict(status='CANDIDATE_PHYSICAL_REVIEW_REQUIRED', parameterization='Canonical composite pairs with whole-mirror handedness and explicit initial prepared grips', sharing_hypothesis='Translated pair costs share parameters as in handoff V1; grip-specific residuals may require later refinement', reachable_product_states=len(parent), parameters=len(keys), initial_rank=initial, final_rank=verified, selected_probes=len(selected), minimum_proven_within_generated_pool=verified == len(keys) and len(selected) == len(keys) - initial, parameter_keys=keys, coverage_scope='Every reachable canonical start/pair context of the candidate graph, not every humanly possible transition', ready_for_mass_timing=False), selected)

def describe_path(graph, ids):
    parts = []
    for identifier in ids:
        e = graph.by_id[identifier]
        a, b = (graph.states[e.source], graph.states[e.target])
        if e.kind == 'MOVE':
            grip = '/'.join((a[h]['grip'] for h in ('left', 'right')))
            part = f'{e.physical_turns}: {e.family}:{e.hand} (L/R thumbs {grip})'
            if e.concurrent != 'NONE':
                part += f' [concurrent {e.concurrent}]'
        else:
            changes = []
            for h in ('left', 'right'):
                if a[h] == b[h]:
                    continue
                if a[h]['grip'] != b[h]['grip']:
                    changes.append(f"{h.upper()} {a[h]['grip']}->{b[h]['grip']}")
                else:
                    changes.append(f"{h.upper()} fingers {a[h]['pose']}->{b[h]['pose']} at {a[h]['grip']}")
            part = e.kind + ': ' + ', '.join(changes)
        parts.append(part)
    return ' ; '.join(parts)

def main():
    root = Path(__file__).resolve().parents[1]
    folder = root / 'data/compiled'
    graph = ExecutionGraph(json.loads((folder / 'candidate_graph.json').read_text()))
    executions = json.loads((folder / 'candidate_executions.json').read_text())
    report, probes = audit_design(graph, executions)
    full, all_probes = reachable_design(graph, executions)
    (folder / 'all_reachable_design_audit.json').write_text(json.dumps(full, indent=2) + '\n')
    (folder / 'all_reachable_candidate_probes.json').write_text(json.dumps(all_probes, indent=2) + '\n')
    with (folder / 'all_reachable_candidate_probes.csv').open('w', newline='') as out:
        writer = csv.writer(out)
        writer.writerow(['Probe', 'Physical turns', 'Exact candidate path', 'End left thumb', 'End right thumb', 'Executable as written', 'Correction if needed', 'Status'])
        for p in all_probes:
            end = graph.states[p['end']]
            writer.writerow([p['id'], p['algorithm'], describe_path(graph, p['edge_ids']), end['left']['grip'], end['right']['grip'], '', '', 'Requires physical review; not released for timing'])
    (folder / 'composite_design_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    (folder / 'candidate_prefix_probes.json').write_text(json.dumps(probes, indent=2) + '\n')
    with (folder / 'candidate_prefix_probes.csv').open('w', newline='') as out:
        writer = csv.writer(out)
        writer.writerow(['Probe', 'Parent TRAIN case', 'Move blocks', 'Physical turns', 'Start', 'Finish left thumb', 'Finish right thumb', 'Status'])
        for p in probes:
            end = graph.states[p['end']]
            writer.writerow([p['id'], p['case_name'], p['move_blocks'], p['algorithm'], 'Both thumbs FRONT; include the same initial regrip', end['left']['grip'], end['right']['grip'], 'Candidate: not released for timing'])
    print(json.dumps({k: report[k] for k in ['training_cases', 'parameters', 'initial_rank', 'rank_deficit', 'selected_probes', 'final_rank', 'minimum_proven_within_prefix_pool', 'ready_for_mass_timing']}))
if __name__ == '__main__':
    main()
