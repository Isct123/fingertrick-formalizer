import hashlib, heapq, json, math
from itertools import count
import numpy as np
from .execution_graph import ExecutionGraph
from .observed_design import composite_key
from .notation_intake import parse_extended
from .cube import permutation

class CompositeGraph:

    def __init__(self, spec):
        if spec.get('schema_version') != 'observed-composite-v1':
            raise ValueError('Expected observed composite specification')
        self.spec = spec
        self.physical = ExecutionGraph(spec['physical_graph'])
        self.states, self.edges, self.by_id = (self.physical.states, self.physical.edges, self.physical.by_id)
        self.moves = {e.id: dict(family=e.family, hand=e.hand, concurrent=e.concurrent, grips_before={h.upper(): self.states[e.source][h]['grip'] for h in ('left', 'right')}, grips_after={h.upper(): self.states[e.target][h]['grip'] for h in ('left', 'right')}) for e in self.edges if e.kind == 'MOVE'}
        self.motifs = {}
        for path in spec['observed_paths']:
            self.physical.validate_path(path['start'], path['edge_ids'], path['algorithm'])
            self._register(path['start'], path['edge_ids'])
            mirrored = [i + ':mirror' for i in path['edge_ids']]
            if all((i in self.by_id for i in mirrored)):
                self._register(self.by_id[mirrored[0]].source, mirrored)
        self.parameter_keys = tuple(sorted({m['key'] for ms in self.motifs.values() for m in ms}))
        self.fingerprint = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()

    def _register(self, start, ids):
        previous = None
        bridge = []
        state = start
        for identifier in ids:
            bridge.append(identifier)
            e = self.by_id[identifier]
            if e.kind != 'MOVE':
                continue
            key = composite_key(self.moves.get(previous), self.moves[identifier])
            motif = dict(ids=tuple(bridge), current=identifier, key=key, target=e.target)
            bucket = self.motifs.setdefault((state, previous), [])
            if motif not in bucket:
                bucket.append(motif)
            state = e.target
            previous = identifier
            bridge = []
        if bridge:
            raise ValueError('Unpriced trailing preparation is not supported')

    def _features(self, ids):
        if not ids:
            raise ValueError('Empty physical execution')
        state = self.by_id[ids[0]].source
        previous = None
        bridge = []
        counts = {}
        for identifier in ids:
            bridge.append(identifier)
            edge = self.by_id[identifier]
            if edge.kind != 'MOVE':
                continue
            match = next((m for m in self.motifs.get((state, previous), []) if m['ids'] == tuple(bridge)), None)
            if match is None:
                raise ValueError('Unobserved preparation/move adjacency; physical review required')
            counts[match['key']] = counts.get(match['key'], 0) + 1
            state = edge.target
            previous = identifier
            bridge = []
        if bridge:
            raise ValueError('Unpriced trailing preparation is not supported')
        return counts

    def feature_matrix(self, paths):
        return np.array([[c.get(k, 0) for k in self.parameter_keys] for c in (self._features(p) for p in paths)], dtype=float).reshape(len(paths), len(self.parameter_keys))

    def validate_path(self, start, ids, algorithm, end=None):
        result = self.physical.validate_path(start, ids, algorithm, end)
        self._features(ids)
        return result

    def _motif_cost(self, motif, weights, previous):
        return weights[motif['key']]

    def search(self, algorithm, start, weights, end=None, max_match_tokens=4):
        if start not in self.states or (end is not None and end not in self.states):
            raise ValueError('Unknown endpoint state')
        if type(max_match_tokens) != int or max_match_tokens < 1:
            raise ValueError('Positive match window required')
        if set(weights) != set(self.parameter_keys) or any((not math.isfinite(v) or v < 0 for v in weights.values())):
            raise ValueError('Supply all composite weights, finite and nonnegative')
        tokens = parse_extended(algorithm)
        n = len(tokens)
        effects = {(i, j): permutation(tokens[i:j]) for i in range(n) for j in range(i + 1, min(n, i + max_match_tokens) + 1)}
        initial = (0, start, None)
        distance = {initial: (0.0, 0)}
        parent = {}
        serial = count()
        queue = [(0.0, 0, next(serial), initial)]
        while queue:
            cost, length, _, node = heapq.heappop(queue)
            if (cost, length) != distance[node]:
                continue
            i, state, previous = node
            if i == n and (end is None or state == end):
                parts = []
                cursor = node
                while cursor != initial:
                    cursor, ids = parent[cursor]
                    parts.append(ids)
                ids = [eid for part in reversed(parts) for eid in part]
                return dict(found=True, cost_s=cost, edge_ids=ids, end=state, graph_fingerprint=self.fingerprint, physical_review_required=True, scope='Recombined observed coarse adjacency motifs')
            for motif in self.motifs.get((state, previous), []):
                edge = self.by_id[motif['current']]
                for j in range(i + 1, min(n, i + max_match_tokens) + 1):
                    if effects[i, j] != edge.effect:
                        continue
                    target = (j, edge.target, edge.id)
                    candidate = (cost + self._motif_cost(motif, weights, previous), length + len(motif['ids']))
                    if candidate < distance.get(target, (math.inf, math.inf)):
                        distance[target] = candidate
                        parent[target] = (node, motif['ids'])
                        heapq.heappush(queue, (*candidate, next(serial), target))
        return dict(found=False, cost_s=None, edge_ids=[], reason='No execution using observed adjacency motifs')

def build_spec(physical_spec, executions):
    if any((e['phase'] != 'TRAIN' for e in executions)):
        raise ValueError('TRAIN paths only')
    return dict(schema_version='observed-composite-v1', physical_graph=physical_spec, observed_paths=[{k: e[k] for k in ('start', 'edge_ids', 'algorithm', 'case_name')} for e in executions], evidence_scope='User-reported habitual TRAIN adjacencies and whole mirrors only', model_assumptions='Canonical coarse poses; composite elapsed costs include preparations and regrips. Novel complete executions require practical review.', requires_physical_review=False)

def load_graph(spec):
    if spec.get('cost_model') == 'family_plus_transition_v1':
        return FamilyTransitionGraph(spec)
    return CompositeGraph(spec) if spec.get('schema_version') == 'observed-composite-v1' else ExecutionGraph(spec)

class FamilyTransitionGraph(CompositeGraph):

    def __init__(self, spec):
        super().__init__(spec)
        self.context_keys = self.parameter_keys
        self.families = tuple(sorted({(e.kind, e.family) for e in self.edges}))
        self.family_keys = tuple(('FAMILY:' + json.dumps(f, separators=(',', ':')) for f in self.families))
        self.family_key_by_pair = dict(zip(self.families, self.family_keys))
        self.parameter_keys = ('TIMER_INTERCEPT',) + self.family_keys + self.context_keys

    def feature_matrix(self, paths):
        rows = []
        for path in paths:
            counts = self._features(path)
            family = dict.fromkeys(self.family_keys, 0)
            for eid in path:
                e = self.by_id[eid]
                family[self.family_key_by_pair[e.kind, e.family]] += 1
            rows.append([1.0] + [family[k] for k in self.family_keys] + [counts.get(k, 0) for k in self.context_keys])
        return np.asarray(rows, dtype=float).reshape(len(paths), len(self.parameter_keys))

    def _motif_cost(self, motif, weights, previous):
        result = weights[motif['key']] + (weights['TIMER_INTERCEPT'] if previous is None else 0.0)
        for eid in motif['ids']:
            e = self.by_id[eid]
            result += weights[self.family_key_by_pair[e.kind, e.family]]
        return result
