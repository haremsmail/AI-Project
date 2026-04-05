from board import Board
"""this page is for vacum cleaner problem using depth first search algorithm to find the path from vacuum to dirt and calculate the cost of the path"""

"""" used find the path using depth first search algorithm  """
class Move:
    """" au class direction movement cost dadozetaua"""
    """ ema deyn pathaka dadozyanauay la regay depth first search alogrithm"""
    """Represents a single move"""
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    """" nauakana  lagal techuy har julakay  lagal  chon pega dagore katek dajule"""
    COSTS = {
        UP: 2,
        DOWN: 0,
        LEFT: 1,
        RIGHT: 1
    }
    
    DIRECTIONS = {
        UP: (-1, 0),
        DOWN: (1, 0),
        LEFT: (0, -1),
        RIGHT: (0, 1)
    }


class VacuumDFS:
    """DFS solver for vacuum cleaner problem"""
    
    def __init__(self, board):
        self.board = board
        self.visited = set()
        self.path = []
        self.total_cost = 0
        self.solution_found = False
        
    def solve(self):
        """Solve using DFS - returns (success, path, cost, message)"""
        start_pos = self.board.vacuum_pos
        self.visited = set()
        self.path = []
        self.total_cost = 0
        
        # Try to find path using DFS
        success = self._dfs(start_pos)
        
        if success:
            self.solution_found = True
            return True, self.path, self.total_cost, "Solution found!"
        else:
            return False, [], 0, "No solution because of obstacles"
    
    def _dfs(self, current_pos):
        """DFS recursive function"""
        # Check if we reached the dirt
        if current_pos == self.board.dirt_pos:
            return True
        
        # Mark as visited
        self.visited.add(current_pos)
        
        # Try all four directions
        for move_name in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            row, col = current_pos
            dr, dc = Move.DIRECTIONS[move_name]
            new_row, new_col = row + dr, col + dc
            new_pos = (new_row, new_col)
            
            # Check if position is valid and not visited
            if (self.board.is_valid_position(new_row, new_col) and 
                new_pos not in self.visited):
                
                # Add to path and cost
                self.path.append({
                    'move': move_name,
                    'from': current_pos,
                    'to': new_pos,
                    'cost': Move.COSTS[move_name]
                })
                self.total_cost += Move.COSTS[move_name]
                
                # Recursive call
                if self._dfs(new_pos):
                    return True
                
                # Backtrack
                self.path.pop()
                self.total_cost -= Move.COSTS[move_name]
        
        return False
    
    def get_path_steps(self):
        """Get readable path steps"""
        steps = []
        for i, move_info in enumerate(self.path, 1):
            step = f"{i}. Move {move_info['move']} from {move_info['from']} to {move_info['to']} (Cost: {move_info['cost']})"
            steps.append(step)
        return steps
