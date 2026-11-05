from dataclasses import dataclass, replace
from enum import Enum
from itertools import product

class Grip(str, Enum):
    FRONT = 'FRONT'
    TOP = 'TOP'
    BOTTOM = 'BOTTOM'

class Pose(str, Enum):
    NEUTRAL = 'NEUTRAL'
    U_BACK = 'U_BACK'
    U_FRONT = 'U_FRONT'
    F_A = 'F_A'
    F_B = 'F_B'
    D_A = 'D_A'
    D_B = 'D_B'
    U2_A = 'U2_A'
    U2_B = 'U2_B'
    D2_A = 'D2_A'
    D2_B = 'D2_B'
    FP_A = 'FP_A'
    FP_B = 'FP_B'

class Side(str, Enum):
    LEFT = 'LEFT'
    RIGHT = 'RIGHT'

class EdgeType(str, Enum):
    MOVE = 'MOVE'
    PREP = 'PREP'
    REGRIP = 'REGRIP'

@dataclass(frozen=True)
class HandState:
    grip: Grip
    pose: Pose

    def __post_init__(self):
        object.__setattr__(self, 'grip', Grip(self.grip))
        object.__setattr__(self, 'pose', Pose(self.pose))

@dataclass(frozen=True)
class State:
    left: HandState
    right: HandState

    def hand(self, side):
        return self.left if Side(side) == Side.LEFT else self.right

    def with_hand(self, side, hand):
        return replace(self, **{'left' if Side(side) == Side.LEFT else 'right': hand})

    def mirror(self):
        return State(self.right, self.left)

    def to_dict(self):
        return {side: [getattr(self, side).grip.value, getattr(self, side).pose.value] for side in ('left', 'right')}

    @classmethod
    def from_dict(cls, value):
        if set(value) != {'left', 'right'}:
            raise ValueError('State must explicitly contain left and right [grip, pose]')
        return cls(HandState(*value['left']), HandState(*value['right']))

@dataclass(frozen=True)
class Move:
    face: str
    turns: int = 1

    def __post_init__(self):
        if self.face not in tuple('UDFBRL') or type(self.turns) is not int or self.turns not in (1, 2, 3):
            raise ValueError(f'Invalid move: {self.face!r}, {self.turns!r}')

    def __str__(self):
        return self.face + {1: '', 2: '2', 3: "'"}[self.turns]
LEGACY_POSES = tuple((p for p in Pose if p not in (Pose.FP_A, Pose.FP_B)))

def raw_states(poses=LEGACY_POSES):
    hands = [HandState(g, p) for g, p in product(Grip, poses)]
    return tuple((State(l, r) for l, r in product(hands, repeat=2)))
