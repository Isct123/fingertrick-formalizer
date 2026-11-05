from .notation_intake import parse_extended
PROTOCOL = 'initial-regrip-included-opening-x-setup-v2'

def split_timing_boundary(algorithm, fused_opening=False):
    tokens = parse_extended(algorithm)
    n = 0
    while not fused_opening and n < len(tokens) and (tokens[n].symbol == 'x'):
        n += 1
    return dict(protocol=PROTOCOL, setup_algorithm=' '.join((t.raw for t in tokens[:n])), timed_algorithm=' '.join((t.raw for t in tokens[n:])), full_algorithm=algorithm, fused_opening=fused_opening, rotation_finish_grips={'LEFT': 'FRONT', 'RIGHT': 'FRONT'})
