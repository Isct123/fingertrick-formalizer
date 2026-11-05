import re
from .model import Move
FAMILIES = ('UN', 'UOHp', 'UNp', 'FT', 'FTp', 'DR', 'DRp', 'R', 'Rp', 'U2', 'D2')
KNOWN_FAMILIES = FAMILIES + ('FP', 'R2', 'UM', 'U4', 'D4', 'R2_RELEASE', 'UTM')

def parse_algorithm(text):
    moves, i = ([], 0)
    while i < len(text):
        if text[i].isspace():
            i += 1
            continue
        match = re.match("([UDFBRL])([2']?)", text[i:])
        if not match:
            raise ValueError(f'Unsupported cube notation at character {i + 1}: {text[i:i + 12]!r}; use outer-face turns only (no slices, wide moves, rotations or groups)')
        moves.append(Move(match[1], {'': 1, '2': 2, "'": 3}[match[2]]))
        i += len(match[0])
    return tuple(moves)

def parse_families(text):
    result = tuple(text.split())
    for i, token in enumerate(result):
        if token not in KNOWN_FAMILIES:
            raise ValueError(f'Unknown physical family {token!r} at token {i + 1}')
    return result
