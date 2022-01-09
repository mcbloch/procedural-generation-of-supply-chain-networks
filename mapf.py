# from collections import deque

# from line_profiler import LineProfiler

# from binarytree import tree, bst, heap, Node
from typing import Set, List, Tuple, Optional
from heapq import heappop, heappush
from random import randint, seed

# profile = LineProfiler()
import bitarray.util

from grid import Grid
from models import GridPos, Move, Constraint, Solution

"""
An implementation of a constraint based search algorithm without the concept of time.
This is because agents are paths that stay at their position.
"""

# --------------------
# Different test cases
# --------------------

TEST_CASE = 3

match TEST_CASE:
    case 0:
        # Big working with and without tunnels
        seed(15)
        AGENT_COUNT = 5
        MIN_DISTANCE = 80
        COLS = 20
        ROWS = 20
        MIN_X = 2
        MIN_Y = 2
        SELECT_RANDOM = True
    case 1:
        # Big and pretty invalid without tunnels but working with
        seed(30)
        AGENT_COUNT = 8
        MIN_DISTANCE = 80
        COLS = 20
        ROWS = 20
        MIN_X = 4
        MIN_Y = 4
        SELECT_RANDOM = True
    case 2:
        # Small example. Working with and without tunnels
        SELECT_RANDOM = False
        COLS = 8
        ROWS = 8
        INITIAL_AGENTS = [((0, 3), (2, 3)),
                          ((0, 6), (6, 1)),
                          ((5, 6), (1, 1)),
                          ((0, 1), (1, 5))
                          ]
    case 3:
        # Case 2 but with some more agents
        # Needs 13000 iterations
        SELECT_RANDOM = False
        COLS = 8
        ROWS = 8
        INITIAL_AGENTS = [((0, 3), (2, 3)),
                          ((0, 6), (6, 1)),
                          ((5, 6), (1, 1)),
                          ((0, 1), (1, 5)),
                          ((2, 0), (4, 3)),  # (6, 7)),
                          ((2, 7), (3, 1)),
                          ]
    case 4:
        # Horizontal and vertical
        SELECT_RANDOM = False
        COLS = 8
        ROWS = 8
        INITIAL_AGENTS = [((0, 2), (7, 2)),
                          ((0, 4), (7, 4)),
                          ((2, 0), (2, 7)),
                          ((5, 0), (5, 7)),
                          ]

MAX_ITERATIONS = 15000  # 20 000


class Agent:
    def __init__(self, idx, start, end):
        self.idx = idx
        self.start = start
        self.end = end

    def search_consistent_path(
            self, grid: Grid, constraints: Set[Constraint]
    ) -> Optional[List[Tuple[GridPos, Move]]]:
        """
        Search a consistent path for an agent.
        This is a path that satisfies all the agent's constraints.
        """
        path = grid.a_star_algorithm(
            self.start,
            self.end,
            frozenset(c.position for c in constraints if c.agent_idx == self.idx),
        )
        return path


class CTNode:  # (Node)
    """
    A node in our constraint tree

    Is a goal node when the solution is valid.
    Valid => the set of paths have no conflicts.
    """

    def __init__(self, index):
        # A set of constraints imposed on each agent
        self.constraints: frozenset[Constraint] = frozenset()

        # A single consistent solution
        #   = one path for each agent that is consistent with the constraints
        self.solution: Solution = list()

        # Cost of the solution.
        self.cost: int = -1

        self.value = index

    def search_consistent_solution(self):
        """
        Contains only consistent paths.
        Could be invalid despite the fact that the paths are
        consistent with the individual agent constraints,
        due to inter-agent conflicts.

        Once a consistent path is found for every agent, validate these against other agents.
        We do this by simulating their movement along the planned paths (solution).

        If all reach their goal without conflict
            the CTNode is declared the goal node and the solution is returned.
        If a conflict is found <a_i, a_j, v, t?>, the validation halts and the node is declared a non-goal.
            - Generate two new CTNodes as children of N
                adding the <a_i, v> and <a_j, v> constraint to the constraint sets.

        Node: For every node the low level search is only activated for one single agent
            (the one with the extra constraint)
        """
        pass

    def sic(self) -> None:
        """
        Set the cost of this node: sum of individual single-agent costs
        :return:
        """
        self.cost = 0
        for path in self.solution:
            self.cost += len(path)

    def validate_solution(self) -> Optional[Tuple[int, int, GridPos]]:
        """
        Check if the solution is a valid one. If we find a conflict; return the first one.
        Here we have the biggest difference in comparison to an algorith with moving agents
            as we can remove the time constraint.
        """
        # TODO Make better performance
        for j, path in enumerate(self.solution):
            for k, path2 in enumerate(self.solution[(j + 1):]):
                path_blocking = bitarray.util.zeros(COLS * ROWS)
                for (pos, move) in path:
                    path_blocking |= move.blocking_path
                path2_blocking = bitarray.util.zeros(COLS * ROWS)
                for (pos, move) in path2:
                    path2_blocking |= move.blocking_path

                if (path_blocking & path2_blocking).any():
                    # Conflict!
                    conflict_idx = (path_blocking & path2_blocking).index(bitarray.bitarray('1'))
                    return j, (j + 1 + k), GridPos(conflict_idx // ROWS, conflict_idx % ROWS)
        return None


class MAPF:
    """Multi-agent path finding on a grid"""

    def __init__(self, grid: Grid, agents: List[Agent]):
        self.grid = grid
        self.agents: List[Agent] = agents

        self.next_index = -1

    def get_next_index(self) -> int:
        self.next_index += 1
        return self.next_index

    @staticmethod
    def find_best_node(nodes: List[CTNode]) -> CTNode:
        best_node = nodes[0]
        for node in nodes[1:]:
            if node.cost < best_node.cost:
                best_node = node
        return best_node

    def print_constraints(
            self, constraints: Set[Constraint], solution: Solution, agent_idx: int
    ):
        cs = {c.position for c in constraints if c.agent_idx == agent_idx}
        path = solution[agent_idx]
        for row_i in range(self.grid.rows):
            for col_i in range(self.grid.cols):
                curr_pos = GridPos(col_i, row_i)
                has_constraint = curr_pos in cs
                is_on_path = curr_pos in path
                print(
                    "{} ".format(
                        "#" if has_constraint else ("^" if is_on_path else ".")
                    ),
                    end="",
                )
            print()

    def cbs(self) -> Solution:
        """
        *Conflict based search Algorithm* on a Grid without a notion of time

        High level: generate constraints for the different agents
        Low level: find paths for individual agents consistent their respective constraints

        When on the low level paths are found that conflict,
        new constraints are added to resolve one of the conflicts
        and the low-level is invoked again.

        High level: Search the constraint tree (CT)
            Performs a best-first search on CT with nodes ordered by cost

            Constraint tree: Starts with an empty set of constraints.
                A successor node will inherit the parent constraint
                and adds a single new one for a single agent
        """
        open_nodes: List[Tuple[int, int, CTNode]] = []

        # Start with a simple constraint tree
        root = CTNode(self.get_next_index())

        # As a start find all paths of all agents with no constraints
        for agent in self.agents:
            path = agent.search_consistent_path(self.grid, root.constraints)
            if not path:
                print("Invalid start path?")
                return
            root.solution.append(path)
        root.sic()

        heappush(open_nodes, (root.cost, root.value, root))

        iteration = 0
        while len(open_nodes) != 0:
            iteration += 1
            if iteration % 5 == 0:
                print(f"Iteration {iteration}")
            if iteration > MAX_ITERATIONS:
                print("Nothing found")
                return None

            # lowest solution cost
            (Pcost, Pidx, P) = heappop(open_nodes)
            # print("Selecting node with best cost {} having {} constraints".format(P.cost, len(P.constraints)))

            # validate the paths in P until a conflict occurs
            conflict = P.validate_solution()

            # self.print_constraints(P.constraints, P.solution, 0)
            # print("--------")
            # self.print_constraints(P.constraints, P.solution, 1)

            if not conflict:
                return P.solution  # Return our goal

            (ai, aj, c_pos) = conflict
            for agent_idx in [ai, aj]:
                A = CTNode(self.get_next_index())
                A.constraints = P.constraints.union({Constraint(agent_idx, c_pos)})
                A.solution = P.solution.copy()

                # Update the solution in A by invoking the low-level of the agent
                path = self.agents[agent_idx].search_consistent_path(
                    self.grid, A.constraints
                )

                # If we have a valid path on the local scale
                if path:
                    A.solution[agent_idx] = path
                    A.sic()

                    # print("new for {}".format(agent_idx))
                    # self.print_constraints(A.constraints, A.solution, agent_idx)

                    heappush(open_nodes, (A.cost, A.value, A))


def distance(n1: GridPos, n2: GridPos) -> int:
    return pow(n2.x - n1.x, 2) + pow(n2.y - n1.y, 2)


def get_random_path_pair(cols, rows):
    while True:
        fr = GridPos(randint(MIN_X, cols - 3), randint(MIN_Y, rows - 3))
        to = GridPos(randint(MIN_X, cols - 3), randint(MIN_Y, rows - 3))
        if fr != to and distance(fr, to) > MIN_DISTANCE:
            return fr, to


def get_random_path_pairs(count, cols, rows):
    pairs = []
    nodes: Set = set()
    for _ in range(count):
        while True:
            pair = get_random_path_pair(cols, rows)
            if pair[0] not in nodes and pair[1] not in nodes:
                nodes.add(pair[0])
                nodes.add(pair[1])
                pairs.append(pair)
                break
    return pairs


def do_algorithm_thingy() -> Tuple[Grid, Solution, List[Agent]]:
    cols = COLS
    rows = ROWS
    initial_grid = Grid(cols, rows)

    if SELECT_RANDOM:

        initial_agents: List[Agent] = []

        for agent_idx, (fr, to) in enumerate(
                get_random_path_pairs(AGENT_COUNT, cols, rows)
        ):
            initial_agents.append(
                Agent(agent_idx, GridPos(fr.x, fr.y), GridPos(to.x, to.y))
            )
    else:
        initial_agents: List[Agent] = [
            Agent(agent_idx, GridPos(fr[0], fr[1]), GridPos(to[0], to[1])) for (agent_idx, (fr, to)) in
            enumerate(INITIAL_AGENTS)]

    mapf = MAPF(initial_grid, initial_agents)
    optimal_solution = mapf.cbs()

    print("Optimal solution")
    print(optimal_solution)

    return initial_grid, optimal_solution, initial_agents


def search_good_random_seed():
    start_seed = 30
    seed_nr = start_seed
    while True:
        seed(seed_nr)
        print(f"Running for seed {seed_nr}")
        grid, solution, agents = do_algorithm_thingy()
        if solution:
            print(f"Found solution for seed {seed_nr}")
            break
        else:
            seed_nr += 1


if __name__ == "__main__":
    # search_good_random_seed()
    do_algorithm_thingy()
