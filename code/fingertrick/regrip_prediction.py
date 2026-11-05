import numpy as np
from .robust_accuracy import RobustPredictor

def regrip_features(graph, row):
    values = dict(initial_regrips=0.0, internal_regrips=0.0, initial_regrip_hands=0.0, internal_regrip_hands=0.0, turn_active_hand_changes=0.0, turn_other_hand_changes=0.0, turn_both_hand_changes=0.0, preparation_grip_changes=0.0, early_internal_regrips=0.0, late_internal_regrips=0.0, consecutive_regrips=0.0, regrips_before_same_hand=0.0)
    edges = [graph.by_id[i] for i in row['edge_ids']]
    total = sum((e.kind == 'MOVE' for e in edges))
    progress = 0
    for i, e in enumerate(edges):
        changed = [h.upper() for h in ('left', 'right') if graph.states[e.source][h]['grip'] != graph.states[e.target][h]['grip']]
        if e.kind == 'REGRIP':
            prefix = 'initial' if progress == 0 else 'internal'
            values[prefix + '_regrips'] += 1
            values[prefix + '_regrip_hands'] += len(changed)
            if progress:
                values['early_internal_regrips'] += progress <= total / 2
                values['late_internal_regrips'] += progress > total / 2
            values['consecutive_regrips'] += i > 0 and edges[i - 1].kind == 'REGRIP'
            nextmove = next((m for m in edges[i + 1:] if m.kind == 'MOVE'), None)
            values['regrips_before_same_hand'] += nextmove is not None and (e.hand == nextmove.hand or 'BOTH' in (e.hand, nextmove.hand))
        elif e.kind == 'PREP':
            values['preparation_grip_changes'] += len(changed)
        elif e.kind == 'MOVE':
            if e.hand == 'BOTH':
                values['turn_both_hand_changes'] += len(changed)
            else:
                values['turn_active_hand_changes'] += sum((h == e.hand for h in changed))
                values['turn_other_hand_changes'] += sum((h != e.hand for h in changed))
            progress += 1
    return values

class RegripPredictor(RobustPredictor):

    def __init__(self, graph, rows):
        super().__init__(graph, rows)
        self.regrip_values = [regrip_features(graph, r) for r in rows]
        self.rg = np.array([list(v.values()) for v in self.regrip_values])
        self.mode = False
        self.rgcache = {}

    def kernel(self, train, mix=0.0):
        if not self.mode:
            return super().kernel(train, mix)
        base = super().kernel(train, 0.2)
        key = tuple(train)
        if key not in self.rgcache:
            x = self.rg / self.n[:, None]
            scale = np.maximum(np.std(x[train], axis=0), 1e-08)
            x = x / scale
            dd = np.maximum(0, (x * x).sum(1)[:, None] + (x * x).sum(1)[None, :] - 2 * x @ x.T)
            td = dd[np.ix_(train, train)]
            positive = td[td > 1e-10]
            bandwidth = max(1.0, float(np.median(positive))) if len(positive) else 1.0
            self.rgcache[key] = np.exp(-dd / bandwidth)
        return (1 - mix) * base + mix * self.rgcache[key]

    def predict(self, config, train, test):
        old = self.mode
        self.mode = config['method'] == 'huber_regrip'
        try:
            return super().predict(config, train, test)
        finally:
            self.mode = old

def configurations():
    from .robust_accuracy import configurations as original
    menu = [c for c in original() if c['method'] in ('count_reference', 'rate_reference', 'huber_action_hybrid')]
    return menu + [dict(method='huber_regrip', alpha=a, delta=d, mix=m) for a in (1.0, 100.0) for d in (0.03, 0.08) for m in (0.1, 0.3)]
