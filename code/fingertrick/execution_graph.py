from dataclasses import dataclass
from collections import defaultdict
import hashlib, json, heapq, math
from itertools import count
from .cube import permutation, commute, IDENTITY
from .notation_intake import parse_extended

@dataclass(frozen=True)
class ExecutionEdge:
    id: str
    source: str
    target: str
    kind: str
    physical_turns: str
    parameter_key: str
    family: str
    hand: str
    concurrent: str
    evidence: str

    @property
    def effect(self):
        return permutation(self.physical_turns)

class ExecutionGraph:

    def __init__(self, spec, allow_synthetic=False):
        if spec.get('schema_version') != 2:
            raise ValueError('Expected execution graph schema 2')
        if spec.get('synthetic', False) and (not allow_synthetic):
            raise ValueError('Synthetic fixture is not research evidence')
        self.spec = spec
        self.states = spec['states']
        self.edges = []
        self.outgoing = defaultdict(list)
        for name, s in self.states.items():
            for side in ('left', 'right'):
                h = s[side]
                if h['grip'] not in ('FRONT', 'TOP', 'BOTTOM', 'RELEASED') or not h['pose'] or h['pose'] in ('UNKNOWN', '?'):
                    raise ValueError(f'Explicit grip and known pose required: {name}')
                if h['grip'] == 'RELEASED' and (not h.get('release_evidence')):
                    raise ValueError('RELEASED is an observed thumb-contact state, not a wildcard grip')
            if not s.get('evidence'):
                raise ValueError('Every physical state requires evidence')
        ids = set()
        for raw in spec['edges']:
            if raw.get('status') != 'CONFIRMED':
                raise ValueError('Only explicitly confirmed edges can enter physical search')
            e = ExecutionEdge(**{k: raw[k] for k in ExecutionEdge.__dataclass_fields__})
            if e.id in ids or e.source not in self.states or e.target not in self.states:
                raise ValueError('Duplicate edge or missing state')
            ids.add(e.id)
            if not e.evidence or not e.parameter_key or e.hand not in ('LEFT', 'RIGHT', 'BOTH', 'NONE'):
                raise ValueError('Edge requires evidence, parameter key, and explicit hand')
            a, b = (self.states[e.source], self.states[e.target])
            grip_change = any((a[h]['grip'] != b[h]['grip'] for h in ('left', 'right')))
            tokens = parse_extended(e.physical_turns)
            if e.kind in ('PREP', 'REGRIP'):
                if tokens or e.source == e.target or (e.kind == 'REGRIP') != grip_change:
                    raise ValueError('Invalid PREP/REGRIP state change')
                if e.concurrent != 'NONE':
                    raise ValueError('Concurrent changes belong to the enclosing MOVE')
            elif e.kind == 'MOVE':
                if not tokens:
                    raise ValueError('MOVE requires physical turns')
                if e.concurrent not in ('NONE', 'PREP', 'REGRIP', 'TURNS', 'TURNS_AND_REGRIP'):
                    raise ValueError('Unknown concurrency annotation')
                if e.hand in ('LEFT', 'RIGHT') and e.concurrent in ('PREP', 'REGRIP'):
                    support = 'right' if e.hand == 'LEFT' else 'left'
                    changed = a[support]['grip'] != b[support]['grip']
                    if (e.concurrent == 'REGRIP') != changed:
                        raise ValueError('Concurrent PREP/REGRIP must match support-hand grip change')
                if e.concurrent in ('TURNS', 'TURNS_AND_REGRIP'):
                    if len(tokens) < 2 or any((not commute(a.canonical(), b.canonical()) for i, a in enumerate(tokens) for b in tokens[i + 1:])):
                        raise ValueError('Simultaneous turns require pairwise commuting logical actions')
                    affected = [{i for i, j in enumerate(permutation(t.canonical())) if i != j} for t in tokens]
                    if any((x & y for i, x in enumerate(affected) for y in affected[i + 1:])):
                        raise ValueError('Simultaneous independent turns must act on disjoint layers')
            else:
                raise ValueError('Unknown edge kind')
            self.edges.append(e)
            self.outgoing[e.source].append(e)
        self.by_id = {e.id: e for e in self.edges}
        self.parameter_keys = tuple(sorted({e.parameter_key for e in self.edges}))
        self.fingerprint = hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

    def validate_path(self, start, edge_ids, algorithm, end=None):
        if start not in self.states:
            raise ValueError('Unknown start state')
        state = start
        turns = []
        for name in edge_ids:
            e = self.by_id[name]
            if e.source != state:
                raise ValueError(f'Physical discontinuity before {name}')
            state = e.target
            if e.kind == 'MOVE':
                turns.append(e.physical_turns)
        if permutation(' '.join(turns)) != permutation(algorithm):
            raise ValueError('Physical path does not equal exact logical algorithm')
        if end is not None and end != state:
            raise ValueError('Wrong physical end state')
        return state

    def search(self, algorithm, start, weights, end=None, max_match_tokens=4):
        if start not in self.states or (end is not None and end not in self.states):
            raise ValueError('Unknown endpoint state')
        if type(max_match_tokens) != int or max_match_tokens < 1:
            raise ValueError('Positive match window required')
        if set(weights) != set(self.parameter_keys) or any((not math.isfinite(v) or v < 0 for v in weights.values())):
            raise ValueError('Supply all graph weights, finite and nonnegative')
        tokens = parse_extended(algorithm)
        n = len(tokens)
        effects = {(i, j): permutation(tokens[i:j]) for i in range(n) for j in range(i + 1, min(n, i + max_match_tokens) + 1)}
        initial = (0, start)
        dist = {initial: 0.0}
        prev = {}
        serial = count()
        queue = [(0.0, next(serial), initial)]
        while queue:
            cost, _, node = heapq.heappop(queue)
            if cost != dist[node]:
                continue
            i, state = node
            if i == n and (end is None or state == end):
                path = []
                cursor = node
                while cursor != initial:
                    before, e = prev[cursor]
                    path.append(e.id)
                    cursor = before
                return dict(found=True, cost_s=cost, edge_ids=list(reversed(path)), end=state, graph_fingerprint=self.fingerprint, max_match_tokens=max_match_tokens)
            for e in self.outgoing[state]:
                advances = [i] if e.kind != 'MOVE' else [j for j in range(i + 1, min(n, i + max_match_tokens) + 1) if effects[i, j] == e.effect]
                for j in advances:
                    target = (j, e.target)
                    candidate = cost + weights[e.parameter_key]
                    if candidate < dist.get(target, math.inf):
                        dist[target] = candidate
                        prev[target] = (node, e)
                        heapq.heappush(queue, (candidate, next(serial), target))
        return dict(found=False, cost_s=None, edge_ids=[], reason='No path in explicit graph and configured match window')

def with_whole_mirrors(spec):
    from copy import deepcopy
    from .cube import reflect_algorithm
    result = deepcopy(spec)
    for name, state in spec['states'].items():
        key = name + ':mirror'
        if key in result['states']:
            raise ValueError('Mirror state ID collision')
        result['states'][key] = dict(state, left=deepcopy(state['right']), right=deepcopy(state['left']), evidence=state['evidence'] + '; whole-hand mirror convention')
    ids = {e['id'] for e in spec['edges']}
    for edge in spec['edges']:
        mirrored = dict(edge, id=edge['id'] + ':mirror', source=edge['source'] + ':mirror', target=edge['target'] + ':mirror', physical_turns=reflect_algorithm(edge['physical_turns']), hand={'LEFT': 'RIGHT', 'RIGHT': 'LEFT', 'BOTH': 'BOTH', 'NONE': 'NONE'}[edge['hand']], evidence=edge['evidence'] + '; whole-hand mirror convention')
        if mirrored['id'] in ids:
            raise ValueError('Mirror edge ID collision')
        result['edges'].append(mirrored)
    return result
