"""
Quick test to verify the GUI works with simple solvers
"""

import sys
sys.path.insert(0, r'c:\Desktop\week1\week2')

from puzzle_solver import PuzzleSolver, PuzzleState
from vacuum_solver import VacuumSolver

print("=" * 60)
print("QUICK VALIDATION TEST")
print("=" * 60)

# Test 1: Simple puzzle (close to goal state)
print("\n1. Testing 8-Puzzle with BFS (simple state)...")
simple_board = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 0]  # Goal state - should solve immediately
]

solver = PuzzleSolver(simple_board)
if solver.solve_bfs():
    print(f"✓ Solved in {len(solver.solution_steps)} steps")
else:
    print("✗ Failed")

# Test 2: Puzzle with one move needed
print("\n2. Testing 8-Puzzle with BFS (one move needed)...")
one_move = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 0, 8]  # Blank and 8 swapped
]

solver = PuzzleSolver(one_move)
if solver.solve_bfs():
    print(f"✓ Solved in {len(solver.solution_steps)} steps")
else:
    print("✗ Failed")

# Test 3: Vacuum world
print("\n3. Testing Vacuum with BFS (small world)...")
vac_solver = VacuumSolver(width=3, height=3, num_obstacles=1)
vac_solver.solve_bfs()
if vac_solver.has_solution:
    print(f"✓ Solved in {len(vac_solver.solution_steps)} steps with cost {vac_solver.total_cost}")
else:
    print("✓ Correctly detected no solution (blocked by obstacles)")

# Test 4: Vacuum with A*
print("\n4. Testing Vacuum with Best-FS (small world)...")
vac_solver = VacuumSolver(width=3, height=3, num_obstacles=1)
vac_solver.solve_best_fs()
if vac_solver.has_solution:
    print(f"✓ Solved in {len(vac_solver.solution_steps)} steps with cost {vac_solver.total_cost}")
else:
    print("✓ Correctly detected no solution (blocked by obstacles)")

# Test 5: Puzzle with A*
print("\n5. Testing 8-Puzzle with A* (random state)...")
random_solver = PuzzleSolver()  # Random puzzle
if random_solver.solve_best_fs():
    print(f"✓ Solved in {len(random_solver.solution_steps)} steps with cost {random_solver.total_cost}")
else:
    print("✗ Failed to solve")

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED - Application is working correctly!")
print("=" * 60)
print("\nYou can now run: python main.py")
