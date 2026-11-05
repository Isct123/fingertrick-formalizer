import numpy as np
from scipy.optimize import minimize_scalar

def mean_operator(features, counts, variances):
    x = np.asarray(features, float)
    n = np.asarray(counts, float)
    v = np.asarray(variances, float)
    if np.any(n < 0) or np.any(v <= 0):
        raise ValueError('Invalid counts or variances')
    w = n / v
    information = x.T @ (w[:, None] * x)
    if np.linalg.matrix_rank(information) != x.shape[1]:
        raise ValueError('Design does not identify the model')
    h = x @ np.linalg.solve(information, x.T * w)
    return h

def risk_components(operator, counts, variances):
    h = np.asarray(operator, float)
    n = np.asarray(counts, float)
    v = np.asarray(variances, float)
    if np.any(np.abs(h[:, n == 0]) > 1e-08):
        raise ValueError('Estimator uses unmeasured cells')
    noise = np.divide(v, n, out=np.zeros_like(v), where=n > 0)
    variance = h * h @ noise
    amplification = np.abs(h - np.eye(len(n))).sum(1)
    return (variance, amplification)

def worst_risk(operator, counts, variances, departure=0.0):
    variance, amplification = risk_components(operator, counts, variances)
    return float(np.max(variance + departure ** 2 * amplification ** 2))

def best_blend(operator, counts, variances, departure):
    if np.any(np.asarray(counts) <= 0):
        raise ValueError('Direct means require every cell to be measured')
    h = np.asarray(operator, float)
    identity = np.eye(len(counts))
    objective = lambda t: worst_risk((1 - t) * h + t * identity, counts, variances, departure)
    result = minimize_scalar(objective, bounds=(0, 1), method='bounded', options={'xatol': 1e-10})
    t = min((0.0, float(result.x), 1.0), key=objective)
    return (t, objective(t))

def grid_blend_formula(a, b, repetitions, sigma2, departure):
    h = (a + b - 1) / (a * b)
    q = 1 - h
    c = 4 * q
    noise = sigma2 / repetitions
    aa = noise * q
    bb = departure ** 2 * c * c
    weight = bb / (aa + bb) if aa + bb else 0.0
    risk = noise * h + (aa * bb / (aa + bb) if aa + bb else 0.0)
    return (weight, risk)
