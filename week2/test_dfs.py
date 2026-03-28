"""
Test script for Vacuum Cleaner DFS Solver
Demonstrates algorithm working without GUI
"""

from board import Board
from vacuum_dfs import VacuumDFS
from solution_writer import SolutionWriter


def test_vacuum_dfs():
    """Test the DFS solver with a sample board"""
    print("=" * 60)
    print("VACUUM CLEANER - DFS SOLVER TEST")
    print("=" * 60)
    
    # Generate a random board
    print("\n1. Generating random 6x6 board...")
    board = Board(width=6, height=6)
    board.generate_random(obstacle_count=8)
    
    print(f"   Board generated successfully!")
    print(f"   Vacuum at: {board.vacuum_pos}")
    print(f"   Dirt at: {board.dirt_pos}")
    print(f"\n   Initial Board:")
    print("   " + board.to_string().replace("\n", "\n   "))
    
    # Solve using DFS
    print("\n2. Solving using DFS algorithm...")
    solver = VacuumDFS(board)
    success, path, cost, message = solver.solve()
    
    if success:
        print(f"   ✅ Solution found!")
        print(f"   Total moves: {len(path)}")
        print(f"   Total cost: {cost}")
        
        # Show first 5 steps
        print(f"\n   Solution steps (first 5):")
        steps = solver.get_path_steps()
        for step in steps[:5]:
            print(f"   {step}")
        if len(steps) > 5:
            print(f"   ... and {len(steps) - 5} more steps")
    else:
        print(f"   ❌ No solution found: {message}")
    
    # Write to file
    print("\n3. Writing solution to solution.txt...")
    writer = SolutionWriter("solution.txt")
    writer.write(board, path, cost, success, solver)
    print("   ✅ Solution written successfully!")
    
    # Read and display part of the file
    print("\n4. Reading solution file...")
    content = writer.read_solution()
    if content:
        lines = content.split('\n')
        print("\n   First 15 lines of solution.txt:")
        for line in lines[:15]:
            print(f"   {line}")
        print("   ...")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    test_vacuum_dfs()
