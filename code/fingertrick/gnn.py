import copy, hashlib, json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .learning import clean_records, grouped_split, incidence, metrics, PROTOCOL

def graph_tensors(graph):
    if hasattr(graph, 'motifs'):
        g, encoder = graph_tensors(graph.physical)
        edge_index = {e.id: i for i, e in enumerate(graph.edges)}
        contexts = sorted({(previous, m['current'], m['key']) for (_, previous), ms in graph.motifs.items() for m in ms}, key=str)
        g['context_previous'] = torch.tensor([edge_index[p] if p else 0 for p, _, _ in contexts])
        g['context_current'] = torch.tensor([edge_index[c] for _, c, _ in contexts])
        g['context_start'] = torch.tensor([p is None for p, _, _ in contexts], dtype=torch.float32).reshape(-1, 1)
        g['groups'] = torch.tensor([graph.parameter_keys.index(k) for _, _, k in contexts])
        g['nkeys'] = len(graph.parameter_keys)
        encoder['contexts'] = contexts
        return (g, encoder)
    states = sorted(graph.states)
    index = {k: i for i, k in enumerate(states)}
    fields = ('left.grip', 'left.pose', 'right.grip', 'right.pose')

    def value(s, f):
        h, k = f.split('.')
        return s[h][k]
    categories = {f: sorted({value(s, f) for s in graph.states.values()}) for f in fields}
    nodes = [[float(value(graph.states[name], f) == v) for f in fields for v in categories[f]] for name in states]
    edge_fields = ('kind', 'family', 'hand', 'concurrent', 'physical_turns')
    ec = {f: sorted({getattr(e, f) for e in graph.edges}) for f in edge_fields}
    features = [[float(getattr(e, f) == v) for f in edge_fields for v in ec[f]] for e in graph.edges]
    sources = [index[e.source] for e in graph.edges]
    targets = [index[e.target] for e in graph.edges]
    groups = [graph.parameter_keys.index(e.parameter_key) for e in graph.edges]
    return (dict(node=torch.tensor(nodes, dtype=torch.float32), edge=torch.tensor(features, dtype=torch.float32), source=torch.tensor(sources), target=torch.tensor(targets), groups=torch.tensor(groups), nkeys=len(graph.parameter_keys)), dict(states=states, node_categories=categories, edge_categories=ec))

class CostGNN(nn.Module):

    def __init__(self, node_dim, edge_dim, hidden=24, layers=2, composite=False):
        super().__init__()
        self.input = nn.Linear(node_dim, hidden)
        self.message = nn.ModuleList((nn.Linear(hidden + edge_dim, hidden) for _ in range(layers)))
        self.update = nn.ModuleList((nn.Linear(2 * hidden, hidden) for _ in range(layers)))
        self.readout = nn.Sequential(nn.Linear(4 * hidden + 2 * edge_dim + 1 if composite else 2 * hidden + edge_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, g):
        h = F.relu(self.input(g['node']))
        src, dst = (g['source'], g['target'])
        degree = torch.zeros(len(h), 1, device=h.device).index_add_(0, dst, torch.ones(len(dst), 1, device=h.device)).clamp_min(1)
        for message, update in zip(self.message, self.update):
            m = F.relu(message(torch.cat((h[src], g['edge']), dim=1)))
            aggregate = torch.zeros_like(h).index_add_(0, dst, m) / degree
            h = F.relu(update(torch.cat((h, aggregate), dim=1)))
        descriptors = torch.cat((h[src], h[dst], g['edge']), dim=1)
        if 'context_current' in g:
            descriptors = torch.cat((descriptors[g['context_previous']] * (1 - g['context_start']), descriptors[g['context_current']], g['context_start']), dim=1)
        edge_cost = F.softplus(self.readout(descriptors).flatten()) + 1e-06
        total = torch.zeros(g['nkeys'], device=h.device).index_add_(0, g['groups'], edge_cost)
        count = torch.zeros_like(total).index_add_(0, g['groups'], torch.ones_like(edge_cost))
        return total / count.clamp_min(1)

def fit_gnn(graph, records, epochs=300, hidden=24, layers=2, seed=2026, learning_rate=0.01, validation_fraction=0.2):
    clean = clean_records(graph, records)
    train, val = grouped_split(clean, validation_fraction, seed)
    if type(epochs) != int or epochs < 1 or hidden < 1 or (layers < 1) or (not np.isfinite(learning_rate)) or (learning_rate <= 0):
        raise ValueError('Invalid GNN hyperparameters')
    torch.manual_seed(seed)
    g, encoder = graph_tensors(graph)
    model = CostGNN(g['node'].shape[1], g['edge'].shape[1], hidden, layers, composite=hasattr(graph, 'motifs'))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    x = torch.tensor(incidence(graph, [r['edge_ids'] for r in train]), dtype=torch.float32)
    y = torch.tensor([r['time_s'] for r in train], dtype=torch.float32)
    vx = torch.tensor(incidence(graph, [r['edge_ids'] for r in val]), dtype=torch.float32)
    vy = torch.tensor([r['time_s'] for r in val], dtype=torch.float32)
    best = float('inf')
    best_state = None
    best_epoch = None
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        costs = model(g)
        loss = F.mse_loss(x @ costs, y)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite GNN training loss')
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            vl = F.mse_loss(vx @ model(g), vy).item()
        if vl < best:
            best = vl
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        weights = model(g).numpy().astype(float)
    artifact = dict(model='physical_graph_message_passing', protocol=PROTOCOL, graph_fingerprint=graph.fingerprint, measurement_method=clean[0].get('measurement_method'), weights_s=dict(zip(graph.parameter_keys, weights.tolist())), seed=seed, hidden=hidden, layers=layers, epochs=epochs, best_epoch=best_epoch, encoder=encoder, state_dict={k: v.tolist() for k, v in best_state.items()}, training_case_names=sorted({r['case_name'] for r in train}), validation_case_names=sorted({r['case_name'] for r in val}), training_metrics=metrics(y.numpy(), x.numpy() @ weights), validation_metrics=metrics(vy.numpy(), vx.numpy() @ weights), training_record_hash=hashlib.sha256(json.dumps(clean, sort_keys=True).encode()).hexdigest(), warning='Validation selects epoch using TRAIN cases only. Individual edge costs are not established by predictive fit.')
    return artifact
