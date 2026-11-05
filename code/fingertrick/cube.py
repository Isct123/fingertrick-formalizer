from functools import lru_cache
from itertools import product
from .notation_intake import parse_extended
NORMALS = {'R': (1, 0, 0), 'L': (-1, 0, 0), 'U': (0, 1, 0), 'D': (0, -1, 0), 'F': (0, 0, 1), 'B': (0, 0, -1)}
STICKERS = tuple(((p, n) for p in product((-1, 0, 1), repeat=3) for n in NORMALS.values() if sum((a * b for a, b in zip(p, n))) == 1))
INDEX = {s: i for i, s in enumerate(STICKERS)}
IDENTITY = tuple(range(54))

def rotate(v, axis, quarters):
    for _ in range(quarters % 4):
        x, y, z = v
        v = ((x, -z, y), (z, y, -x), (-y, x, z))[axis]
    return v

def compose(first, second):
    return tuple((second[first[i]] for i in range(54)))

@lru_cache(maxsize=128)
def turn(symbol, turns):
    if symbol in 'xyz':
        axis = 'xyz'.index(symbol)
        layers = (-1, 0, 1)
        q = -turns
    elif symbol in ('M', 'E', 'S'):
        axis = {'M': 0, 'E': 1, 'S': 2}[symbol]
        layers = (0,)
        q = turns if symbol in ('M', 'E') else -turns
    else:
        n = NORMALS[symbol[0]]
        axis = next((i for i, a in enumerate(n) if a))
        sign = n[axis]
        layers = (0, sign) if symbol.endswith('w') else (sign,)
        q = -sign * turns
    return tuple((INDEX[rotate(p, axis, q), rotate(n, axis, q)] if p[axis] in layers else i for i, (p, n) in enumerate(STICKERS)))

def permutation(algorithm):
    tokens = parse_extended(algorithm) if isinstance(algorithm, str) else algorithm
    p = IDENTITY
    for t in tokens:
        p = compose(p, turn(t.symbol, t.turns))
    return p

def equivalent(a, b):
    return permutation(a) == permutation(b)

def commute(a, b):
    pa, pb = (permutation(a), permutation(b))
    return compose(pa, pb) == compose(pb, pa)
REFLECT = tuple((INDEX[(-p[0], p[1], p[2]), (-n[0], n[1], n[2])] for p, n in STICKERS))

def reflected_permutation(p):
    return compose(compose(REFLECT, p), REFLECT)

def reflect_algorithm(algorithm):
    symbols = ('R', 'L', 'U', 'D', 'F', 'B', 'Rw', 'Lw', 'Uw', 'Dw', 'Fw', 'Bw', 'M', 'E', 'S', 'x', 'y', 'z')
    result = []
    for t in parse_extended(algorithm):
        effect = reflected_permutation(turn(t.symbol, t.turns))
        matches = [s + {1: '', 2: '2', 3: "'"}[q] for s in symbols for q in (1, 2, 3) if turn(s, q) == effect]
        if len(matches) != 1:
            raise ValueError('Reflection must have unique canonical token')
        mirrored = matches[0]
        if t.turns == 2:
            quarter = reflected_permutation(turn(t.symbol, 1))
            target_symbol = mirrored[:-1]
            sign = 1 if turn(target_symbol, 1) == quarter else -1
            mirrored = target_symbol + ("2'" if sign * t.directed_quarters < 0 else '2')
        result.append(mirrored)
    return ' '.join(result)
