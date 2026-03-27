"""
Vacuum World Solver using different search algorithms
"""

import random
from typing import List, Tuple
from search_algorithms import DepthFirstSearch, BreadthFirstSearch, BestFirstSearch


class VacuumState:
    """Represents the state of a vacuum world"""
    
    # Constants for board elements
    EMPTY = 0
    DIRT = 1
    OBSTACLE = 2
    VACUUM = 3
    VACUUM_ON_DIRT = 4
    
    def __init__(self, board: List[List[int]], vacuum_pos: Tuple[int, int]):
        self.board = board
        self.vacuum_pos = vacuum_pos
    
    def __eq__(self, other):
        if isinstance(other, VacuumState):
            return self.board == other.board and self.vacuum_pos == other.vacuum_pos
        return False
    
    def __hash__(self):
        return hash((tuple(tuple(row) for row in self.board), self.vacuum_pos))
    
    def copy(self):
        """Create a deep copy of the state"""
        return VacuumState(
            [row[:] for row in self.board],
            tuple(self.vacuum_pos)
        )
    
    def is_goal(self, goal_pos: Tuple[int, int]) -> bool:
        """Check if vacuum is on dirt"""
        row, col = self.vacuum_pos
        if row < 0 or row >= len(self.board) or col < 0 or col >= len(self.board[0]):
            return False
        return self.board[row][col] == self.DIRT
    
    def get_neighbors(self) -> List[Tuple['VacuumState', int]]:
        """Get all valid neighboring states with move costs"""
        neighbors = []
        row, col = self.vacuum_pos
        
        # Moves: up (cost 2), down (cost 0), left (cost 1), right (cost 1)
        moves = [
            (-1, 0, 2),  # up - cost 2
            (1, 0, 0),   # down - cost 0
            (0, -1, 1),  # left - cost 1
            (0, 1, 1)    # right - cost 1
        ]
        
        height = len(self.board)
        width = len(self.board[0])
        
        for dr, dc, cost in moves:
            new_row, new_col = row + dr, col + dc
            
            # Check bounds
            if 0 <= new_row < height and 0 <= new_col < width:
                # Check if not an obstacle
                if self.board[new_row][new_col] != self.OBSTACLE:
                    new_state = self.copy()
                    new_state.vacuum_pos = (new_row, new_col)
                    neighbors.append((new_state, cost))
        
        return neighbors
    
    def manhattan_distance_to(self, target: Tuple[int, int]) -> int:
        """Calculate Manhattan distance to target dirt"""
        row, col = self.vacuum_pos
        target_row, target_col = target
        return abs(row - target_row) + abs(col - target_col)


class VacuumSolver:
    """Solver for Vacuum World using different search algorithms"""
    
    def __init__(self, width: int = 4, height: int = 4, num_obstacles: int = 3):
        self.width = width
        self.height = height
        self.num_obstacles = num_obstacles
        self.board = None
        self.initial_state = None
        self.dirt_pos = None
        self.vacuum_pos = None
        self.solution_steps = []
        self.total_cost = 0
        self.algorithm_name = ""
        self.has_solution = True
        
        self._generate_random_world()
    
    def _generate_random_world(self):
        """Generate a random vacuum world"""
        self.board = [[VacuumState.EMPTY for _ in range(self.width)] for _ in range(self.height)]
        
        # Place obstacles
        obstacles_placed = 0
        while obstacles_placed < self.num_obstacles:
            row = random.randint(0, self.height - 1)
            col = random.randint(0, self.width - 1)
            if self.board[row][col] == VacuumState.EMPTY:
                self.board[row][col] = VacuumState.OBSTACLE
                obstacles_placed += 1
        
        # Place dirt (not on obstacle)
        while True:
            row = random.randint(0, self.height - 1)
            col = random.randint(0, self.width - 1)
            if self.board[row][col] == VacuumState.EMPTY:
                self.board[row][col] = VacuumState.DIRT
                self.dirt_pos = (row, col)
                break
        
        # Place vacuum (not on obstacle or dirt)
        while True:
            row = random.randint(0, self.height - 1)
            col = random.randint(0, self.width - 1)
            if self.board[row][col] == VacuumState.EMPTY:
                self.vacuum_pos = (row, col)
                break
        
        self.initial_state = VacuumState([row[:] for row in self.board], self.vacuum_pos)
    
    def solve_dfs(self) -> bool:
        """Solve using Depth-First Search"""
        self.algorithm_name = "Depth-First Search (DFS)"
        
        def goal_test(state):
            return state.is_goal(self.dirt_pos)
        
        def get_neighbors(state):
            return state.get_neighbors()
        
        def get_cost(state1, state2):
            return 1
        
        solver = DepthFirstSearch(
            self.initial_state,
            goal_test,
            get_neighbors,
            get_cost
        )
        
        if solver.solve():
            self.solution_steps = solver.solution
            self.total_cost = solver.total_cost
            self.has_solution = True
            return True
        
        self.has_solution = False
        return False
    
    def solve_bfs(self) -> bool:
        """Solve using Breadth-First Search"""
        self.algorithm_name = "Breadth-First Search (BFS)"
        
        def goal_test(state):
            return state.is_goal(self.dirt_pos)
        
        def get_neighbors(state):
            return state.get_neighbors()
        
        def get_cost(state1, state2):
            return 1
        
        solver = BreadthFirstSearch(
            self.initial_state,
            goal_test,
            get_neighbors,
            get_cost
        )
        
        if solver.solve():
            self.solution_steps = solver.solution
            self.total_cost = solver.total_cost
            self.has_solution = True
            return True
        
        self.has_solution = False
        return False
    
    def solve_best_fs(self) -> bool:
        """Solve using Best-First Search with Manhattan distance heuristic"""
        self.algorithm_name = "Best-First Search"
        
        def goal_test(state):
            return state.is_goal(self.dirt_pos)
        
        def get_neighbors(state):
            return state.get_neighbors()
        
        def get_cost(state1, state2):
            return 1
        
        def heuristic(state):
            return state.manhattan_distance_to(self.dirt_pos)
        
        solver = BestFirstSearch(
            self.initial_state,
            goal_test,
            get_neighbors,
            get_cost,
            heuristic
        )
        
        if solver.solve():
            self.solution_steps = solver.solution
            self.total_cost = solver.total_cost
            self.has_solution = True
            return True
        
        self.has_solution = False
        return False
    
    def get_solution_steps(self):
        """Get all states in the solution path"""
        steps = [self.initial_state]
        for state, _ in self.solution_steps:
            steps.append(state)
        return steps
    
    def save_solution(self, filename: str = "solution.txt"):
        """Save solution to file"""
        with open(filename, 'w') as f:
            f.write(f"Vacuum World Solver - {self.algorithm_name}\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Move Costs:\n")
            f.write("  Up: 2, Down: 0, Left: 1, Right: 1\n\n")
            
            f.write("Initial State:\n")
            f.write(self._format_board(self.initial_state.board, self.initial_state.vacuum_pos))
            f.write("\n\n")
            
            if not self.has_solution:
                f.write("NO SOLUTION FOUND!\n")
                f.write("The vacuum cannot reach the dirt because of obstacles.\n")
                return
            
            f.write("Solution Steps:\n")
            f.write("-" * 50 + "\n")
            
            for idx, (next_state, cost) in enumerate(self.solution_steps, 1):
                f.write(f"\nStep {idx} (Move Cost: {cost}):\n")
                f.write(self._format_board(next_state.board, next_state.vacuum_pos))
            
            f.write("\n\n" + "=" * 50 + "\n")
            f.write(f"Total Steps: {len(self.solution_steps)}\n")
            f.write(f"Total Cost: {self.total_cost}\n")
            f.write(f"Algorithm: {self.algorithm_name}\n")
    
    @staticmethod
    def _format_board(board: List[List[int]], vacuum_pos: Tuple[int, int]) -> str:
        """Format board for display"""
        symbols = {
            VacuumState.EMPTY: ".",
            VacuumState.DIRT: "D",
            VacuumState.OBSTACLE: "#",
            VacuumState.VACUUM: "V"
        }
        
        result = ""
        for i, row in enumerate(board):
            for j, cell in enumerate(row):
                if (i, j) == vacuum_pos:
                    result += "V "
                else:
                    result += symbols.get(cell, "?") + " "
            result += "\n"
        return result
