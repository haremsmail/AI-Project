"""
Solution writer for Vacuum Cleaner problem
"""
import os
from datetime import datetime


class SolutionWriter:
    """Writes solution to file"""
    
    def __init__(self, filename="solution.txt"):
        self.filename = filename
    
    def write(self, board, path, total_cost, solution_found, solver):
        """Write solution to file"""
        with open(self.filename, 'w') as f:
            # Header
            f.write("=" * 50 + "\n")
            f.write("VACUUM CLEANER SEARCH PROBLEM SOLUTION\n")
            f.write("Algorithm: Depth First Search (DFS)\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")
            
            # Initial Board
            f.write("INITIAL BOARD:\n")
            f.write(board.to_string() + "\n")
            f.write(f"\nBoard Size: {board.height} x {board.width}\n")
            f.write(f"Vacuum Position: {board.vacuum_pos}\n")
            f.write(f"Dirt Position: {board.dirt_pos}\n\n")
            
            # Solution
            if solution_found:
                f.write("SOLUTION FOUND!\n")
                f.write("-" * 50 + "\n\n")
                
                # Steps
                f.write("STEPS TO SOLUTION:\n")
                steps = solver.get_path_steps()
                for step in steps:
                    f.write(step + "\n")
                
                f.write("\n" + "-" * 50 + "\n")
                f.write(f"TOTAL COST: {total_cost}\n")
                f.write(f"TOTAL MOVES: {len(path)}\n")
                
            else:
                f.write("NO SOLUTION FOUND!\n")
                f.write("-" * 50 + "\n")
                f.write("There is no solution because of obstacles\n")
            
            f.write("\n" + "=" * 50 + "\n")
    
    def read_solution(self):
        """Read solution from file"""
        if os.path.exists(self.filename):
            with open(self.filename, 'r') as f:
                return f.read()
        return None
