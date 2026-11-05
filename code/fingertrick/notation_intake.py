from dataclasses import dataclass
import re

@dataclass(frozen=True)
class Token:
    raw: str
    kind: str
    symbol: str
    turns: int

    @property
    def directed_quarters(self):
        return -2 if self.raw.endswith("2'") else -1 if self.turns == 3 else self.turns

    def canonical(self):
        return self.symbol + {1: '', 2: '2', 3: "'"}[self.turns]

def parse_extended(text):
    result = []
    i = 0
    while i < len(text):
        if text[i].isspace():
            i += 1
            continue
        match = re.match("([URFDLB]w|[URFDLBurfdlbxyzMES])((?:2'?|')?)", text[i:])
        if not match:
            raise ValueError(f'Unsupported notation at character {i + 1}')
        raw = match[0]
        symbol = match[1]
        if symbol in ('x', 'y', 'z'):
            kind = 'ROTATION'
        elif symbol in ('M', 'E', 'S'):
            kind = 'SLICE'
        elif symbol.endswith('w') or symbol in 'urfdlb':
            kind = 'WIDE'
            symbol = symbol[0].upper() + 'w'
        else:
            kind = 'OUTER'
        result.append(Token(raw, kind, symbol, {'': 1, '2': 2, "2'": 2, "'": 3}[match[2]]))
        i += len(raw)
    return tuple(result)
