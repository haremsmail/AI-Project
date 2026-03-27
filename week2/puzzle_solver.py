"""
8-Puzzle Solver using different search algorithms
"""

import random
import copy
from typing import List, Tuple
from search_algorithms import DepthFirstSearch, BreadthFirstSearch, BestFirstSearch


class PuzzleState:
    """Represents the state of an 8-puzzle"""
    
    def __init__(self, board: List[List[int]]):
        self.board = board
    
    def __eq__(self, other):
        if isinstance(other, PuzzleState):
            return self.board == other.board
        return self.board == other
    
    def __hash__(self):
        return hash(tuple(tuple(row) for row in self.board))
    
    def copy(self):
        return PuzzleState([row[:] for row in self.board])
    
    @staticmethod
    def goal_state():
        """Return the goal state for 8-puzzle"""
        return [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
    
    def is_goal(self):
        """Check if this is the goal state"""
        return self.board == self.goal_state()
    
    def find_blank(self) -> Tuple[int, int]:
        """Find position of blank (0)"""
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == 0:
                    return i, j
        return None
    
    def get_neighbors(self) -> List[Tuple['PuzzleState', int]]:
        """Get all valid neighboring states"""
        neighbors = []
        blank_row, blank_col = self.find_blank()
        
        # Move blank up, down, left, right
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for dr, dc in moves:
            new_row, new_col = blank_row + dr, blank_col + dc
            
            if 0 <= new_row < 3 and 0 <= new_col < 3:
                new_state = self.copy()
                # Swap blank with neighbor
                new_state.board[blank_row][blank_col] = new_state.board[new_row][new_col]
                new_state.board[new_row][new_col] = 0
                neighbors.append((new_state, 1))  # Each move costs 1
        
        return neighbors
    
    def manhattan_distance(self) -> int:
        """Calculate Manhattan distance heuristic"""
        distance = 0
        goal = self.goal_state()
        
        for i in range(3):
            for j in range(3):
                if self.board[i][j] != 0:
                    value = self.board[i][j]
                    # Find where this value should be
                    goal_row = (value - 1) // 3
                    goal_col = (value - 1) % 3
                    distance += abs(i - goal_row) + abs(j - goal_col)
        
        return distance


class PuzzleSolver:
    """Solver for 8-puzzle using different search algorithms"""
    
    def __init__(self, initial_board: List[List[int]] = None):
        if initial_board is None:
            self.initial_state = self._generate_random_puzzle()
        else:
            self.initial_state = PuzzleState(initial_board)
        
        self.solution_steps = []
        self.total_cost = 0
        self.algorithm_name = ""
    
    def _generate_random_puzzle(self) -> PuzzleState:
        """Generate a random solvable puzzle state"""
        # Start with goal state and shuffle it
        board = [row[:] for row in PuzzleState.goal_state()]
        
        # Make random moves (at least 20) to ensure randomness
        state = PuzzleState(board)
        for _ in range(50):
            neighbors = state.get_neighbors()
            state, _ = random.choice(neighbors)
        
        return state
    
    def solve_dfs(self) -> bool:
        """Solve using Depth-First Search"""
        self.algorithm_name = "Depth-First Search (DFS)"
        
        def goal_test(state):
            return state.is_goal()
        
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
            return True
        return False
    
    def solve_bfs(self) -> bool:
        """Solve using Breadth-First Search"""
        self.algorithm_name = "Breadth-First Search (BFS)"
        
        def goal_test(state):
            return state.is_goal()
        
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
            return True
        return False
    
    def solve_best_fs(self) -> bool:
        """Solve using Best-First Search with Manhattan distance heuristic"""
        self.algorithm_name = "Best-First Search (A*)"
        
        def goal_test(state):
            return state.is_goal()
        
        def get_neighbors(state):
            return state.get_neighbors()
        
        def get_cost(state1, state2):
            return 1
        
        def heuristic(state):
            return state.manhattan_distance()
        
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
            return True
        return False
    
    def get_solution_steps(self) -> List[PuzzleState]:
        """Get all states in the solution path"""
        steps = [self.initial_state]
        for state, _ in self.solution_steps:
            steps.append(state)
        return steps
    
    def save_solution(self, filename: str = "solution.txt"):
        """Save solution to file"""
        with open(filename, 'w') as f:
            f.write(f"8-Puzzle Solver - {self.algorithm_name}\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Initial State:\n")
            f.write(self._format_board(self.initial_state.board))
            f.write("\n\n")
            
            f.write("Solution Steps:\n")
            f.write("-" * 50 + "\n")
            
            current_state = self.initial_state
            for idx, (next_state, cost) in enumerate(self.solution_steps, 1):
                f.write(f"\nStep {idx} (Cost: {cost}):\n")
                f.write(self._format_board(next_state.board))
                current_state = next_state
            
            f.write("\n\n" + "=" * 50 + "\n")
            f.write(f"Total Steps: {len(self.solution_steps)}\n")
            f.write(f"Total Cost: {self.total_cost}\n")
            f.write(f"Algorithm: {self.algorithm_name}\n")
    
    @staticmethod
    def _format_board(board: List[List[int]]) -> str:
        """Format board for display"""
        result = ""
        for row in board:
            result += " ".join(f"{val:2d}" if val != 0 else " ." for val in row) + "\n"
        return result
