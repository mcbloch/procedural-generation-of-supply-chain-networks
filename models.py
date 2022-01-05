from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple


class PathType(Enum):
    EMPTY = " "
    A = "A"
    B = "B"


@dataclass(frozen=True)
class GridPos:
    x: int
    y: int


@dataclass(frozen=True)
class Vec2:
    x: int
    y: int


@dataclass(frozen=True)
class Constraint:
    """
    Contains the information of a constraint over our path.

    Here it only contains an agent and a position indicating
    that an agent can't be at some position.
    """

    agent_idx: int
    position: GridPos


class MoveType(Enum):
    # Idx, length
    NORMAL = (0, 1)
    UNDERGROUND = (1, 4)


@dataclass(frozen=True)
class Move:
    pos: GridPos
    blocks: set[GridPos]
    mtype: MoveType
    weight: int


@dataclass(frozen=True)
class MoveOption:
    offset: Vec2
    blocks: set[Vec2]
    mtype: MoveType
    weight: int


Solution = List[List[Tuple[GridPos, Move]]]
