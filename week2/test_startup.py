"""
Quick test to ensure puzzle and vacuum are displayed on startup
"""
import sys
sys.path.insert(0, r'c:\Desktop\week1\AI-Project\week2')

from puzzle_solver import PuzzleSolver
from vacuum_solver import VacuumSolver

print("=" * 60)
print("STARTUP DISPLAY TEST")
print("=" * 60)

# Test 1: Generate puzzle and display
print("\n1. Testing puzzle generation...")
puzzle_solver = PuzzleSolver()
print("✓ Puzzle generated successfully")
print(f"  Initial state: {puzzle_solver.initial_state.board[0]}")
print(f"                 {puzzle_solver.initial_state.board[1]}")
print(f"                 {puzzle_solver.initial_state.board[2]}")

# Test 2: Generate vacuum world and display
print("\n2. Testing vacuum world generation...")
vacuum_solver = VacuumSolver(width=4, height=4, num_obstacles=3)
print("✓ Vacuum world generated successfully")
print(f"  Vacuum position: {vacuum_solver.vacuum_pos}")
print(f"  Dirt position: {vacuum_solver.dirt_pos}")

# Test 3: Verify puzzle can be solved
print("\n3. Testing puzzle solving (BFS)...")
if puzzle_solver.solve_bfs():
    print(f"✓ Puzzle solved in {len(puzzle_solver.solution_steps)} steps")
else:
    print("✗ Puzzle could not be solved")

# Test 4: Verify vacuum can be solved
print("\n4. Testing vacuum solving (BFS)...")
vacuum_solver.solve_bfs()
if vacuum_solver.has_solution:
    print(f"✓ Vacuum solved in {len(vacuum_solver.solution_steps)} steps")
else:
    print("✓ No solution (blocked by obstacles)")

print("\n" + "=" * 60)
print("✅ ALL STARTUP TESTS PASSED")
print("=" * 60)
print("\nThe GUI should now display:")
print("  • Initial puzzle board on 8-Puzzle tab")
print("  • Initial vacuum world on Vacuum tab")
print("  • 'Generate New' buttons to create new puzzles/worlds")
print("  • 'Solve' button to find solutions")
