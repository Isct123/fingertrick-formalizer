from collections import Counter
import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import minimize
from .graph_subset_prediction import Predictor

def joint_fit(kernel, base, target, weights, alpha):
    covariance = kernel + alpha * np.diag(1 / np.maximum(weights, 1e-12))
    ib = np.linalg.solve(covariance, base)
    beta = np.linalg.lstsq(base.T @ ib, base.T @ np.linalg.solve(covariance, target), rcond=None)[0]
    coefficient = np.linalg.solve(covariance, target - base @ beta)
    return (beta, coefficient)

def polish_huber(kernel, base, target, obs_path, weights, alpha, delta, beta, coefficient):
    eigen, vectors = np.linalg.eigh(kernel)
    positive = eigen > 1e-10
    eigen = eigen[positive]
    vectors = vectors[:, positive]
    root = np.sqrt(eigen)
    scale = max(float(np.mean(base[:, 1])), 1.0)
    baseline = base.copy()
    baseline[:, 1] /= scale
    phi = vectors * root
    design = np.c_[baseline, phi]
    penalty = np.diag(np.r_[0.0, 0.0, np.full(len(eigen), alpha)])
    totals = np.bincount(obs_path, weights, minlength=len(base))
    hessian = design.T @ (totals[:, None] * design) + penalty
    chol = cholesky(hessian, lower=True)
    white = solve_triangular(chol, design.T, lower=True).T
    inverse = solve_triangular(chol, np.eye(len(chol)), lower=True)
    wp = inverse @ penalty @ inverse.T
    initial = np.r_[beta[0], beta[1] * scale, root * (vectors.T @ coefficient)]
    initial = chol.T @ initial

    def objective(value):
        residual = target - (white @ value)[obs_path]
        absolute = abs(residual)
        loss = np.where(absolute <= delta, 0.5 * residual ** 2, delta * (absolute - 0.5 * delta))
        gradient = -(white.T @ np.bincount(obs_path, weights * np.clip(residual, -delta, delta), minlength=len(base))) + wp @ value
        return (float(weights @ loss + 0.5 * value @ wp @ value), gradient)
    before = objective(initial)[0]
    result = minimize(objective, initial, jac=True, method='L-BFGS-B', options=dict(maxiter=500, gtol=1e-08, ftol=1e-13, maxls=40))
    theta = solve_triangular(chol.T, result.x, lower=False)
    final_beta = np.array([theta[0], theta[1] / scale])
    final_coefficient = vectors @ (theta[2:] / root)
    after, gradient = objective(result.x)
    if after > before + 1e-09:
        raise ValueError('Convex polishing increased training objective')
    return (final_beta, final_coefficient, dict(success=bool(result.success), iterations=int(result.nit), objective_before=before, objective_after=after, gradient_inf=float(np.max(abs(gradient)))))

class RobustPredictor:

    def __init__(self, graph, rows):
        self.rows = rows
        self.reference = Predictor(graph, rows)
        self.groups = self.reference.groups
        self.n = self.reference.n
        self.x = np.array([list(r['description']['values'].values()) for r in rows])
        self.hx = np.array([list(r['values'].values()) for r in self.reference.hand_descriptions])
        pairs = [(i, t['time_s']) for i, r in enumerate(rows) for t in r['trials']]
        self.obs_id = np.array([p[0] for p in pairs])
        self.obs_time = np.array([p[1] for p in pairs])
        self.kernel_cache = {}
        self.cache = {}
        self.convergence = []

    def kernel(self, train, mix=0.0):
        key = (tuple(train), mix)
        if key in self.kernel_cache:
            return self.kernel_cache[key]
        a = self.x[:, 2:] / self.n[:, None]
        scale = np.maximum(np.std(a[train], axis=0), 1e-08)
        a = a / scale
        dd = np.maximum(0, np.sum(a * a, axis=1)[:, None] + np.sum(a * a, axis=1)[None, :] - 2 * a @ a.T)
        td = dd[np.ix_(train, train)]
        bandwidth = max(1.0, float(np.median(td[td > 1e-10]))) if np.any(td > 1e-10) else 1.0
        result = (1 - mix) * np.exp(-dd / bandwidth) + mix * self.reference.walk_kernel
        self.kernel_cache[key] = result
        return result

    def observations(self, train):
        mask = np.isin(self.obs_id, train)
        ids = self.obs_id[mask]
        times = self.obs_time[mask]
        counts = Counter(self.groups[ids])
        weights = np.array([1 / counts[self.groups[i]] for i in ids])
        return (ids, times, weights)

    def predict(self, config, train, test):
        train = np.asarray(train, dtype=int)
        test = np.asarray(test, dtype=int)
        key = (tuple(sorted(config.items())), tuple(train), tuple(test))
        if key in self.cache:
            return self.cache[key]
        method = config['method']
        alpha = config.get('alpha', 1.0)
        if method in ('rate_reference', 'count_reference'):
            pred = self.reference.predict(dict(method='rate_all' if method == 'rate_reference' else 'move_count', alpha=alpha), train, test)
        elif method.startswith('boost_'):
            from sklearn.ensemble import HistGradientBoostingRegressor
            ids, times, weights = self.observations(train)
            features = self.hx if method == 'boost_hands_absolute' else self.x
            log = method == 'boost_log_rate'
            target = np.log(times / self.n[ids]) if log else times
            model = HistGradientBoostingRegressor(loss='absolute_error', learning_rate=0.05, max_iter=100, max_leaf_nodes=config['leaves'], min_samples_leaf=24, l2_regularization=config['l2'], early_stopping=False, random_state=20261103)
            model.fit(features[ids], target, sample_weight=weights)
            pred = model.predict(features[test])
            if log:
                pred = self.n[test] * np.exp(pred)
        else:
            ids, times, raw_weights = self.observations(train)
            local = np.full(len(self.rows), -1, dtype=int)
            local[train] = np.arange(len(train))
            oi = local[ids]
            kernel = self.kernel(train, config.get('mix', 0.0))
            log = method == 'log_rate'
            if log:
                target = np.log(times / self.n[ids])
                base = np.c_[np.ones(len(train)), 1 / self.n[train]]
                tbase = np.c_[np.ones(len(test)), 1 / self.n[test]]
                kk = kernel[np.ix_(train, train)]
                cross = kernel[np.ix_(test, train)]
            else:
                target = times
                base = np.c_[np.ones(len(train)), self.n[train]]
                tbase = np.c_[np.ones(len(test)), self.n[test]]
                kk = kernel[np.ix_(train, train)] * self.n[train, None] * self.n[None, train]
                cross = kernel[np.ix_(test, train)] * self.n[test, None] * self.n[None, train]
            weights = raw_weights.copy()
            prior = None
            converged = False
            for iteration in range(60 if method.startswith('huber') else 1):
                total = np.bincount(oi, weights, minlength=len(train))
                average = np.bincount(oi, weights * target, minlength=len(train)) / total
                beta, coefficient = joint_fit(kk, base, average, total, alpha)
                fitted = base @ beta + kk @ coefficient
                if prior is not None and np.max(abs(fitted - prior)) < 1e-06:
                    converged = True
                    break
                prior = fitted.copy()
                if method.startswith('huber'):
                    error = target - fitted[oi]
                    weights = raw_weights * np.minimum(1.0, config['delta'] / np.maximum(abs(error), 1e-12))
            polish = None
            if method.startswith('huber') and (not converged):
                beta, coefficient, polish = polish_huber(kk, base, target, oi, raw_weights, alpha, config['delta'], beta, coefficient)
            pred = tbase @ beta + cross @ coefficient
            if log:
                pred = self.n[test] * np.exp(pred)
            if method.startswith('huber'):
                self.convergence.append(dict(method=method, iterations=iteration + 1, converged=converged, polish=polish))
        if not np.isfinite(pred).all():
            raise ValueError('Nonfinite robust prediction')
        self.cache[key] = pred
        return pred

def configurations():
    configs = [dict(method=m, alpha=a) for m in ('rate_reference', 'count_reference', 'squared_raw_rate', 'log_rate') for a in (0.01, 1.0, 100.0)]
    for delta in (0.03, 0.08, 0.15):
        for alpha in (0.01, 1.0, 100.0):
            configs.append(dict(method='huber_rate', delta=delta, alpha=alpha))
    for delta in (0.03, 0.08):
        for alpha in (1.0, 100.0):
            configs.append(dict(method='huber_action_hybrid', delta=delta, alpha=alpha, mix=0.2))
    for method in ('boost_absolute', 'boost_hands_absolute', 'boost_log_rate'):
        for leaves in (3, 7):
            for l2 in (1.0, 10.0):
                configs.append(dict(method=method, leaves=leaves, l2=l2))
    return configs
