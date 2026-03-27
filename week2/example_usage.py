"""
Example usage of the Puzzle and Vacuum solvers
Demonstrates programmatic usage without GUI
"""

from puzzle_solver import PuzzleSolver
from vacuum_solver import VacuumSolver


def demo_puzzle():
    """Demonstrate 8-puzzle solving"""
    print("=" * 60)
    print("8-PUZZLE SOLVER DEMO")
    print("=" * 60)
    
    # Example 1: DFS
    print("\n1. Solving puzzle with Depth-First Search...")
    solver_dfs = PuzzleSolver()
    if solver_dfs.solve_dfs():
        print(f"✓ Solved in {len(solver_dfs.solution_steps)} steps with cost {solver_dfs.total_cost}")
        solver_dfs.save_solution("puzzle_dfs_demo.txt")
        print("  Saved to: puzzle_dfs_demo.txt")
    
    # Example 2: BFS
    print("\n2. Solving puzzle with Breadth-First Search...")
    solver_bfs = PuzzleSolver()
    if solver_bfs.solve_bfs():
        print(f"✓ Solved in {len(solver_bfs.solution_steps)} steps with cost {solver_bfs.total_cost}")
        solver_bfs.save_solution("puzzle_bfs_demo.txt")
        print("  Saved to: puzzle_bfs_demo.txt")
    
    # Example 3: Best-FS (A*)
    print("\n3. Solving puzzle with Best-First Search (A*)...")
    solver_best = PuzzleSolver()
    if solver_best.solve_best_fs():
        print(f"✓ Solved in {len(solver_best.solution_steps)} steps with cost {solver_best.total_cost}")
        solver_best.save_solution("puzzle_best_demo.txt")
        print("  Saved to: puzzle_best_demo.txt")


def demo_vacuum():
    """Demonstrate vacuum world solving"""
    print("\n" + "=" * 60)
    print("VACUUM WORLD SOLVER DEMO")
    print("=" * 60)
    
    # Example 1: DFS
    print("\n1. Solving vacuum world with Depth-First Search...")
    solver_dfs = VacuumSolver(width=4, height=4, num_obstacles=2)
    solver_dfs.solve_dfs()
    if solver_dfs.has_solution:
        print(f"✓ Solved in {len(solver_dfs.solution_steps)} steps with cost {solver_dfs.total_cost}")
        solver_dfs.save_solution("vacuum_dfs_demo.txt")
        print("  Saved to: vacuum_dfs_demo.txt")
    else:
        print("✗ No solution found (blocked by obstacles)")
    
    # Example 2: BFS
    print("\n2. Solving vacuum world with Breadth-First Search...")
    solver_bfs = VacuumSolver(width=4, height=4, num_obstacles=2)
    solver_bfs.solve_bfs()
    if solver_bfs.has_solution:
        print(f"✓ Solved in {len(solver_bfs.solution_steps)} steps with cost {solver_bfs.total_cost}")
        solver_bfs.save_solution("vacuum_bfs_demo.txt")
        print("  Saved to: vacuum_bfs_demo.txt")
    else:
        print("✗ No solution found (blocked by obstacles)")
    
    # Example 3: Best-FS
    print("\n3. Solving vacuum world with Best-First Search...")
    solver_best = VacuumSolver(width=4, height=4, num_obstacles=2)
    solver_best.solve_best_fs()
    if solver_best.has_solution:
        print(f"✓ Solved in {len(solver_best.solution_steps)} steps with cost {solver_best.total_cost}")
        solver_best.save_solution("vacuum_best_demo.txt")
        print("  Saved to: vacuum_best_demo.txt")
    else:
        print("✗ No solution found (blocked by obstacles)")


def demo_custom_puzzle():
    """Demonstrate solving a custom puzzle"""
    print("\n" + "=" * 60)
    print("CUSTOM PUZZLE DEMO")
    print("=" * 60)
    
    # Create a custom puzzle state
    custom_board = [
        [1, 2, 3],
        [4, 0, 6],
        [7, 5, 8]
    ]
    
    print("\nCustom Puzzle State:")
    print("  1 2 3")
    print("  4 . 6")
    print("  7 5 8")
    
    solver = PuzzleSolver(custom_board)
    
    print("\nSolving with A* algorithm...")
    if solver.solve_best_fs():
        print(f"✓ Solved in {len(solver.solution_steps)} steps with cost {solver.total_cost}")
        solver.save_solution("puzzle_custom_demo.txt")
        print("  Saved to: puzzle_custom_demo.txt")
    else:
        print("✗ Could not solve this puzzle")


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " SEARCH ALGORITHMS DEMO - Week 2 Project ".center(58) + "║")
    print("╚" + "=" * 58 + "╝")
    
    # Run demos
    demo_puzzle()
    demo_vacuum()
    demo_custom_puzzle()
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETED")
    print("=" * 60)
    print("\nFor GUI usage, run: python main.py")
