import os
""" means check if clik exist"""
from datetime import datetime


class SolutionWriter:
    """Writes solution to file"""
    
    def __init__(self, filename="solution.txt"):
        """ this run when object is created"""
        self.filename = filename
    
    def write(self, board, path, total_cost, solution_found, solver):
        """Write solution to file the solver dfs  solution found true or false have"""
        with open(self.filename, 'w') as f:
            """ open file for writing"""
            # Header
            f.write("=" * 50 + "\n")
            f.write("VACUUM CLEANER SEARCH PROBLEM SOLUTION\n")
            f.write("Algorithm: Depth First Search (DFS)\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")
            
            # Initial Board
            f.write("INITIAL BOARD:\n")
            """ section title + initial board layout"""
            f.write(board.to_string() + "\n")
            """. . V . .
. # . . .
. . . G .
print borad ex"""
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
                    """"  loopek banau codaka dakatu stepakan lanau soluction.txt nyshan dada"""
                
                f.write("\n" + "-" * 50 + "\n")
                f.write(f"TOTAL COST: {total_cost}\n")
                f.write(f"TOTAL MOVES: {len(path)}\n")
                """ count numbe3r of moves"""
                
            else:
                f.write("NO SOLUTION FOUND!\n")
                f.write("-" * 50 + "\n")
                f.write("There is no solution because of obstacles\n")
            
            f.write("\n" + "=" * 50 + "\n")
    
    def read_solution(self):
        """Read solution from file"""
        if os.path.exists(self.filename):
            """ check daka au file haya  agar habu open daka return content daka
            """
            with open(self.filename, 'r') as f:

                return f.read()
        return None
""" agar nabu return none daaka"""










"""
==================================================
VACUUM CLEANER SEARCH PROBLEM SOLUTION
Algorithm: Depth First Search (DFS)
Generated: 2026-04-05 12:30:00

INITIAL BOARD:
. . V . .
. # . . .
. . . G .

Board Size: 6 x 6
Vacuum Position: (1,2)
Dirt Position: (3,3)

SOLUTION FOUND!
--------------------------------------------------

1. Move RIGHT from (1,2) to (1,3)
2. Move DOWN from (1,3) to (2,3)

--------------------------------------------------
TOTAL COST: 3
TOTAL MOVES: 2
==================================================
"""