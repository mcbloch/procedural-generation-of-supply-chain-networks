import functools
from dataclasses import dataclass
from enum import Enum
from heapq import heappush, heappop
from typing import Dict, List, Tuple, Set, Optional, Generator, Iterable

import bitarray
from BitVector import BitVector
import copy

from bitarray import bitarray as bits
from bitarray.util import zeros as bits_zeros, count_and
from bitarray import frozenbitarray

from models import PathType, GridPos, MoveOption, Vec2, MoveType, Move
import numpy as np


class Direction(Enum):
    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3


@dataclass(frozen=True, eq=True)
class NodePair:
    __slots__ = ['node', 'node_parent']
    node: GridPos
    node_parent: GridPos


@dataclass(frozen=True, eq=True, order=True)
class AStarOption:
    __slots__ = ['f_score', 'g_score', 'node', 'node_parent', 'path_mask']
    f_score: int
    g_score: int
    node: GridPos
    node_parent: GridPos
    path_mask: frozenbitarray


@dataclass
class GridView:
    """
    A bitvector representation indicating which connections of the grid are available
    You have for every node (except the last) an edge to the right (positive x | cols) and up (positive y | rows).
    """

    def __init__(self, cols: int, rows: int, nodes=None):
        self.cols = cols
        self.rows = rows
        # self.edge_count_div_2 = ((self.rows * self.cols - 1) + (self.cols * self.rows - 1))
        # self.edge_count = self.edge_count_div_2 * 2
        # self.edges = BitVector(size=self.edge_count)
        # self.edges.reset(1)  # Set all edges as connected by default
        # self.edges: List[bool] = [True for _i in range(self.edge_count)]

        # 0 -> horizontal (going right edges)
        # 1 -> vertical   (going up edges)
        # if edges is not None:
        #     self.edges = edges.copy()
        # else:
        #     self.edges = np.full((2, self.cols, self.rows), True)

        # Saves if a node can still be travelled to in our graph
        if nodes is not None:
            self.nodes = nodes.copy()
        else:
            self.nodes = np.full((self.cols, self.rows), True)

        # TODO performance
        # Could have better performance if I can check edges in a line close to eachother
        # -> keer bitvector per horizontal or vertical line?

    # @staticmethod
    # def get_direction(fr: GridPos, to: GridPos) -> Tuple[GridPos, Direction]:
    #     if (fr.x == to.x) and abs(fr.y - to.y) == 1:
    #         if fr.y < to.y:
    #             return fr, Direction.UP
    #         else:
    #             return to, Direction.UP
    #     elif (fr.y == to.y) and abs(fr.x - to.x) == 1:
    #         if fr.x < to.x:
    #             return fr, Direction.RIGHT
    #         else:
    #             return to, Direction.RIGHT
    #     else:
    #         print("GridView: Invalid edge direction")

    # def set_edge(self, fr: GridPos, to: GridPos, on: bool):
    #     # edge_idx = self.get_edge_index(fr, to)
    #     node_pos, direction = self.get_direction(fr, to)
    #     self.edges[0 if direction == Direction.RIGHT else 1][node_pos.x][node_pos.y] = on
    #
    # def get_edge(self, fr: GridPos, to: GridPos) -> bool:
    #     node_pos, direction = self.get_direction(fr, to)
    #     return self.edges[0 if direction == Direction.RIGHT else 1][node_pos.x][node_pos.y]
    #     # return self.edges[self.get_edge_index(fr, to)]

    def mark_node_taken(self, node: GridPos):
        # if node.x < self.cols - 1:
        #     self.edges[0][node.x][node.y] = False
        # if node.x > 0:
        #     self.edges[0][node.x - 1][node.y] = False
        # if node.y < self.rows - 1:
        #     self.edges[1][node.x][node.y] = False
        # if node.y > 0:
        #     self.edges[1][node.x][node.y - 1] = False
        self.nodes[node.x, node.y] = False

    # @staticmethod
    # def get_edges(fr: GridPos, to: GridPos) -> Tuple[GridPos, Direction, int]:
    #     if (fr.x == to.x) and fr.y != to.y:
    #         if fr.y < to.y:
    #             # UP
    #             start = fr
    #             end = to
    #         else:
    #             # DOWN
    #             start = to
    #             end = fr
    #         return start, Direction.UP, end.y - start.y
    #
    #     elif (fr.y == to.y) and fr.x != to.x:
    #         if fr.x < to.x:
    #             # RIGHT
    #             start = fr
    #             end = to
    #         else:
    #             # LEFT
    #             start = to
    #             end = fr
    #         return start, Direction.RIGHT, end.x - start.x
    #     else:
    #         print("GridView/get_edges: Nodes need to be different direction")

    # def has_edges(self, fr: GridPos, to: GridPos):
    #     start, direction, distance = self.get_edges(fr, to)
    #     match direction:
    #         case Direction.RIGHT:
    #             return all(
    #                 [self.edges[0][start.x + offset][start.y] for offset in range(distance)])
    #         case Direction.UP:
    #             return all(
    #                 [self.edges[1][start.x][start.y + offset] for offset in range(distance)])

    def can_travel_to(self, node: GridPos):
        return self.nodes[node.x, node.y]

    def deep_copy(self):
        grid_copy = GridView(self.cols, self.rows, self.nodes)
        return grid_copy


class Grid:
    def __init__(self, cols, rows):
        self.cols = cols
        self.rows = rows

        self.data = [
            [PathType.EMPTY for _ in range(self.cols)] for _ in range(self.rows)
        ]
        self.edges: List[NodePair] = []
        self.neighbor_dirs: List[MoveOption] = [
            MoveOption(
                offset=Vec2(-1, 0), blocks=set(), mtype=MoveType.NORMAL, weight=1
            ),
            MoveOption(
                offset=Vec2(1, 0), blocks=set(), mtype=MoveType.NORMAL, weight=1
            ),
            MoveOption(
                offset=Vec2(0, -1), blocks=set(), mtype=MoveType.NORMAL, weight=1
            ),
            MoveOption(
                offset=Vec2(0, 1), blocks=set(), mtype=MoveType.NORMAL, weight=1
            ),
            MoveOption(
                offset=Vec2(-4, 0),
                blocks={Vec2(-3, 0), Vec2(-1, 0)},
                mtype=MoveType.UNDERGROUND,
                weight=6,
            ),
            MoveOption(
                offset=Vec2(4, 0),
                blocks={Vec2(3, 0), Vec2(1, 0)},
                mtype=MoveType.UNDERGROUND,
                weight=6,
            ),
            MoveOption(
                offset=Vec2(0, -4),
                blocks={Vec2(0, -3), Vec2(0, -1)},
                mtype=MoveType.UNDERGROUND,
                weight=6,
            ),
            MoveOption(
                offset=Vec2(0, 4),
                blocks={Vec2(0, 3), Vec2(0, 1)},
                mtype=MoveType.UNDERGROUND,
                weight=6,
            ),
        ]

    def get_node_index(self, node: GridPos) -> int:
        return (node.x * self.rows) + node.y

    def new_grid_view(self, nodes: Iterable[GridPos] = None) -> frozenbitarray:
        return frozenbitarray(self.new_grid_view_mutable(nodes))

    def new_grid_view_mutable(self, nodes: Iterable[GridPos] = None) -> bitarray:
        b = bits_zeros(self.cols * self.rows)
        if nodes is not None:
            for node in nodes:
                b[self.get_node_index(node)] = True
        return b

    # def grid_pos_set_to_bitarray(self, grid_positions: Iterable[GridPos]) -> bitarray:
    #     grid_path_view = self.new_grid_view()
    #     for grid_pos in grid_positions:
    #         grid_path_view[self.get_node_index(grid_pos)] = True
    #     return grid_path_view

    @staticmethod
    @functools.lru_cache()
    def distance(n1: GridPos, n2: GridPos) -> int:
        return pow(n2.x - n1.x, 2) + pow(n2.y - n1.y, 2)

    # @profile
    def get_neighbors(self, v: GridPos, constraints: frozenset[GridPos]) -> Iterable[Tuple[GridPos, Move]]:
        constraint_bits = self.new_grid_view(constraints)
        neighs = []
        for move_option in self.neighbor_dirs:
            new_pos = GridPos(
                v.x + move_option.offset.x, v.y + move_option.offset.y
            )
            if (
                    0 <= new_pos.x < self.cols
                    and 0 <= new_pos.y < self.rows
                    and new_pos not in constraints
            ):
                blocking_path = self.new_grid_view(
                    {
                        GridPos(v.x + block.x, v.y + block.y)
                        for block in move_option.blocks.union({move_option.offset})
                    }
                )
                if not (constraint_bits & blocking_path).any():
                    neighs.append((
                        new_pos,
                        Move(
                            pos=new_pos,
                            blocking_path=blocking_path,
                            mtype=move_option.mtype,
                            weight=move_option.weight,
                        ))
                    )

        return neighs

    def get_f_score(self, v: AStarOption, end: GridPos, g: int):
        dist = self.distance(v.node, end)
        return g + dist

    # @profile
    def reconstruct_path(self, s, t, adj, extra_bl=None, should_match: bitarray = None) \
            -> Tuple[bool, int, bits, List[Tuple[GridPos, Move]]]:

        path = []
        # print("##############################")
        # print("##############################")
        blocked = self.new_grid_view_mutable()
        if extra_bl:
            blocked[self.get_node_index(extra_bl)] = True
        success = self._reconstruct_path(s, t, path, blocked, adj, should_match=should_match)

        # Calculate the cost for this path
        g_score = sum([move.weight for (pos, move) in path])
        # grid_view = np.full((self.cols, self.rows), True)
        return success, g_score, blocked, path

    # @profile
    def _reconstruct_path(self, start: GridPos, to: GridPos, path: List[Tuple[GridPos, Move]],
                          blocked_path: bitarray,
                          adj_mapping: Dict[GridPos, Dict[GridPos, Tuple[int, Move]]],
                          should_match: bitarray = None) -> bool:

        # self.print_grid_path(path)
        if start == to:
            s_move = next(iter(adj_mapping[start].values()))[1]
            path.append((start, s_move))
            path.reverse()
            return True

        # to -> n
        # for (adj_node, (_score, move)) in sorted(adj_mapping[to].items(), key=lambda tup: tup[1][0]):
        for (adj_node, (_score, move)) in sorted(adj_mapping[to].items(), key=lambda tup: tup[1][0]):
            # Check if we can still do this move in our current path?
            if should_match is not None and count_and(should_match, move.blocking_path) > 0:
                continue

            if count_and(blocked_path, move.blocking_path) > 0:
                continue

            path.append((to, move))
            blocked_path |= move.blocking_path
            path_successfull = self._reconstruct_path(start, adj_node, path, blocked_path, adj_mapping,
                                                      should_match=should_match)
            if path_successfull:
                return True
            path.pop()
            blocked_path &= ~move.blocking_path

        return False

    # @profile
    def a_star_algorithm(
            self, start: GridPos, stop: GridPos, constraints: frozenset[GridPos]
    ) -> Optional[List[Tuple[GridPos, Move]]]:
        if start in constraints or stop in constraints:
            return None

        # In this open_list is a list of nodes which have been visited, but who's
        # neighbours haven't all been always inspected, It starts off with the start node
        # And closed_lst is a list of nodes which have been visited
        # and who's neighbors have been always inspected
        start_opt = AStarOption(node=start, node_parent=start, path_mask=self.new_grid_view([start]),
                                g_score=0, f_score=self.distance(start, stop))

        open_lst: List[Tuple[int, int, AStarOption]] = []
        heappush(open_lst, (start_opt.f_score, self.get_node_index(start_opt.node), start_opt))
        closed_lst: Set[AStarOption] = set()

        # distance from the starting node from adjacent
        # g_score: Dict[AStarOption, int] = {start_opt: 0}
        # g + estimated cost from the tile to goal
        # f_score: Dict[AStarOption, int] = {start_opt: self.distance(start, stop)}

        # A Dictionary that holds the list of adjacent nodes with the matching and move
        adj_mapping: Dict[GridPos, Dict[GridPos, Tuple[int, Move]]] = {
            start: {start: (0, Move(start, self.new_grid_view({start}), MoveType.NORMAL, 1))}
        }

        while len(open_lst) > 0:
            n_opt: AStarOption = None

            # find the node with the lowest total heuristic value
            if len(open_lst) > 0:
                # Select the lowest f-score
                _score, _idx, n_opt = heappop(open_lst)

            if n_opt is None:
                # print("Path does not exist!")
                return None

            # if the current node is the stop we can create our path
            if n_opt.node == stop:
                success, _score, _gview, reconst_path = self.reconstruct_path(start, n_opt.node, adj_mapping,
                                                                              should_match=n_opt.path_mask)

                # print("Path found: {}".format(reconst_path))
                if success:
                    return reconst_path

            # for all the neighbors of the current node do
            for (m, move) in self.get_neighbors(n_opt.node, constraints):
                # Check if this move would be valid
                # Now check that we have no upper overlap in our path

                # First we will check in our bitmap if we can reuse the best path of our parent
                # If this fails we will try to construct a new best path for this connection
                #   -> Check for the grid_view of (n, n_adj) if we can still go to m

                grid_view_to_n: frozenbitarray = n_opt.path_mask
                if grid_view_to_n[self.get_node_index(m)] == 1:
                    # print("Invalid path, overlap in itself")
                    continue

                grid_view_to_n: frozenbitarray = frozenbitarray(
                    grid_view_to_n | move.blocking_path)
                score_to_n = n_opt.g_score

                m_to_n_g_score: int = score_to_n + move.weight

                # if the current node is not presenting both open_lst and closed_lst
                # add it to open_lst and note n as it's par

                m_opt = AStarOption(node=m, node_parent=n_opt.node, path_mask=grid_view_to_n,
                                    g_score=m_to_n_g_score,
                                    f_score=m_to_n_g_score + self.distance(m, stop))
                if (m_opt.f_score, m_opt) not in open_lst and m_opt not in closed_lst:
                    heappush(open_lst, (m_opt.f_score, self.get_node_index(m_opt.node), m_opt))

                    # AStar looks up the n_score here, but we use the one from the path
                    if m not in adj_mapping:
                        adj_mapping[m] = {}
                    adj_mapping[m][n_opt.node] = (m_to_n_g_score, move)

                    # g_score[m_opt] = m_to_n_g_score
                    # self.set_f_score(m_opt, stop, g_score, f_score)

                # if the pair is already visited
                # check if it's quicker to first visit with our new n-score
                # and if it is, update adjecency data and scores
                # and if the node was in the closed_lst, move it to open_lst
                # else:
                #     if g_score[m_opt] > m_to_n_g_score:
                #
                #         g_score[m_opt] = m_to_n_g_score
                #         self.set_f_score(m_opt, stop, g_score, f_score)
                #
                #         adj_mapping[m][n_opt.node] = (m_to_n_g_score, move)
                #
                #         if m_opt in closed_lst:
                #             closed_lst.remove(m_opt)
                #             open_lst.add(m_opt)

            # remove n from the open_lst, and add it to closed_lst
            # because all of his neighbors were inspected
            # open_lst.remove(n_opt)
            closed_lst.add(n_opt)

        # print("Path does not exist!")
        return None

    def print_grid(self):
        """broke and doesn't show edges"""
        for line in self.data:
            for pt in line:
                print(str(pt) + " ", end="")
            print()
        print("==================")

    @staticmethod
    def get_marker(nodes, i) -> str:
        if i == len(nodes) - 1:
            return "*"
        else:
            fr = nodes[i]
            to = nodes[i + 1]
            if fr.x == to.x and fr.y < to.y: return "v"
            if fr.x == to.x and fr.y > to.y: return "^"
            if fr.x < to.x and fr.y == to.y: return "<"
            if fr.x > to.x and fr.y == to.y: return ">"

    def print_grid_path(self, path: List[Tuple[GridPos, Move]]):
        nodes = [p for (p, m) in path]
        for y in range(self.rows, -1, -1):
            for x in range(self.cols):
                if GridPos(x, y) in nodes:
                    i = nodes.index(GridPos(x, y))
                    m = self.get_marker(nodes, i)
                    print("{} ".format(m), end="")
                else:
                    print(". ", end="")
            print()
        print("==================")

    # def add_path(self, path):
    #     for (fr, to) in zip(path, path[1:]):
    #         self.edges.append((fr, to))
