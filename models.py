from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple

from bitarray import bitarray


class PathType(Enum):
    EMPTY = " "
    A = "A"
    B = "B"


@dataclass(frozen=True, eq=True, order=True)
class GridPos:
    __slots__ = ['x', 'y']
    x: int
    y: int


@dataclass(frozen=True, eq=True)
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
    __slots__ = ['pos', 'blocking_path', 'mtype', 'weight']
    pos: GridPos
    blocking_path: bitarray
    mtype: MoveType
    weight: int


@dataclass(frozen=True)
class MoveOption:
    __slots__ = ['offset', 'blocks', 'mtype', 'weight']
    offset: Vec2
    blocks: set[Vec2]
    mtype: MoveType
    weight: int


Solution = List[List[Tuple[GridPos, Move]]]
