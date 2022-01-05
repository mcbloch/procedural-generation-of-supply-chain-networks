import functools
from typing import Dict, List, Tuple, Set, Optional

from models import PathType, GridPos, MoveOption, Vec2, MoveType, Move


class Grid:
    def __init__(self, cols, rows):
        self.cols = cols
        self.rows = rows

        self.data = [
            [PathType.EMPTY for _ in range(self.cols)] for _ in range(self.rows)
        ]
        self.edges: List[Tuple[GridPos, GridPos]] = []
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

    @staticmethod
    @functools.lru_cache()
    def distance(n1: GridPos, n2: GridPos) -> int:
        return pow(n2.x - n1.x, 2) + pow(n2.y - n1.y, 2)

    # @profile
    def get_neighbors(self, v: GridPos, constraints: frozenset[GridPos]):
        return (
            (
                new_pos,
                Move(
                    pos=new_pos,
                    blocks=blocks,
                    mtype=move_option.mtype,
                    weight=move_option.weight,
                ),
            )
            for move_option in self.neighbor_dirs
            if (
                (
                    new_pos := GridPos(
                        v.x + move_option.offset.x, v.y + move_option.offset.y
                    )
                )
                and 0 <= new_pos.x < self.cols
                and 0 <= new_pos.y < self.rows
                and (
                    blocks := {
                        GridPos(v.x + block.x, v.y + block.y)
                        for block in move_option.blocks.union({move_option.offset})
                    }
                )
                and new_pos not in constraints
                and all(block not in constraints for block in blocks))
        )

    def get_f_score(self, v, end, g, f) -> int:
        if v in f:
            return f[v]
        else:
            dist = self.distance(v, end)
            f[v] = g[v] + dist
            return f[v]

    @staticmethod
    def reconstruct_path(start: GridPos, to: GridPos, par: Dict[GridPos, Tuple[GridPos, Move]]) \
            -> List[Tuple[GridPos, Move]]:
        reconst_path = []
        n = to

        while par[n][0] != n:
            reconst_path.append((n, par[n][1]))
            n = par[n][0]

        reconst_path.append((start, par[start][1]))

        reconst_path.reverse()
        return reconst_path

    def a_star_algorithm(
            self, start: GridPos, stop: GridPos, constraints: frozenset[GridPos]
    ) -> Optional[List[Tuple[GridPos, Move]]]:
        if start in constraints or stop in constraints:
            return None

        # In this open_list is a list of nodes which have been visited, but who's
        # neighbours haven't all been always inspected, It starts off with the start node
        # And closed_lst is a list of nodes which have been visited
        # and who's neighbors have been always inspected
        open_lst: Set[GridPos] = set()
        open_lst.add(start)
        closed_lst: Set[GridPos] = set()

        # distance from the starting node
        g_score: Dict[GridPos, int] = {start: 0}
        # g + estimated cost from the tile to goal
        f_score: Dict[GridPos, int] = {start: self.distance(start, stop)}

        # par contains an adjac mapping of all nodes
        par: Dict[GridPos, Tuple[GridPos, Move]] = {
            start: (start, Move(start, {start}, MoveType.NORMAL, 1))
        }

        while len(open_lst) > 0:
            n = None

            # find the node with the lowest total heuristic value
            for v in open_lst:
                if n is None or self.get_f_score(
                        v, stop, g_score, f_score
                ) < self.get_f_score(n, stop, g_score, f_score):
                    n = v

            if n is None:
                # print("Path does not exist!")
                return None

            # if the current node is the stop
            # then we start again from start
            if n == stop:
                reconst_path = self.reconstruct_path(start, n, par)

                # print("Path found: {}".format(reconst_path))
                return reconst_path

            # for all the neighbors of the current node do
            for (m, move) in self.get_neighbors(n, constraints):
                # Check if this move would be valid
                # Now check that we have no upper overlap in our path
                valid = True
                # TODO Cache this?
                path = self.reconstruct_path(start, n, par)
                for (_, path_move) in path:
                    if m in path_move.blocks:
                        valid = False
                        # print("Invalid path, overlap in itself")
                        break
                if not valid:
                    continue

                # if the current node is not presenting both open_lst and closed_lst
                # add it to open_lst and note n as it's par
                if m not in open_lst and m not in closed_lst:
                    open_lst.add(m)
                    par[m] = (n, move)
                    g_score[m] = g_score[n] + move.weight
                    self.get_f_score(m, stop, g_score, f_score)  # Update f_score cache

                # otherwise, check if it's quicker to first visit n, then m
                # and if it is, update par data and poo data
                # and if the node was in the closed_lst, move it to open_lst
                else:
                    if g_score[m] > g_score[n] + move.weight:
                        g_score[m] = g_score[n] + move.weight
                        del f_score[m]  # Invalidate f_score cache
                        par[m] = (n, move)

                        if m in closed_lst:
                            closed_lst.remove(m)
                            open_lst.add(m)

            # remove n from the open_lst, and add it to closed_lst
            # because all of his neighbors were inspected
            open_lst.remove(n)
            closed_lst.add(n)

        # print("Path does not exist!")
        return None

    def print_grid(self):
        """broke and doesn't show edges"""
        for line in self.data:
            for pt in line:
                print(str(pt) + " ", end="")
            print()
        print("==================")

    def add_path(self, path):
        for (fr, to) in zip(path, path[1:]):
            self.edges.append((fr, to))
