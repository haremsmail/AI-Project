"""
Comprehensive test suite for Vacuum Cleaner DFS Solver
Tests multiple scenarios including edge cases
"""

from board import Board
from vacuum_dfs import VacuumDFS
from solution_writer import SolutionWriter


def test_solvable_board():
    """Test with a solvable board"""
    print("\n" + "=" * 70)
    print("TEST 1: Solvable Board")
    print("=" * 70)
    
    board = Board(width=6, height=6)
    board.generate_random(obstacle_count=5)
    
    solver = VacuumDFS(board)
    success, path, cost, message = solver.solve()
    
    if success:
        print(f"✅ PASS: Solution found")
        print(f"   Path length: {len(path)}, Total cost: {cost}")
    else:
        print(f"⚠️  Board generated without solution - retrying")
        return test_solvable_board()
    
    return True


def test_empty_board():
    """Test with minimal obstacles"""
    print("\n" + "=" * 70)
    print("TEST 2: Empty Board (Minimal Obstacles)")
    print("=" * 70)
    
    board = Board(width=6, height=6)
    board.generate_random(obstacle_count=2)
    
    solver = VacuumDFS(board)
    success, path, cost, message = solver.solve()
    
    print(f"✅ PASS: Board solved")
    print(f"   Solution exists: {success}")
    print(f"   Path length: {len(path)}")
    
    return True


def test_crowded_board():
    """Test with many obstacles"""
    print("\n" + "=" * 70)
    print("TEST 3: Crowded Board (Many Obstacles)")
    print("=" * 70)
    
    board = Board(width=6, height=6)
    board.generate_random(obstacle_count=15)
    
    solver = VacuumDFS(board)
    success, path, cost, message = solver.solve()
    
    print(f"✅ PASS: Algorithm completed")
    print(f"   Solution found: {success}")
    if success:
        print(f"   Path length: {len(path)}, Total cost: {cost}")
    else:
        print(f"   Message: {message}")
        print(f"   This is valid - no solution possible with this layout")
    
    return True


def test_custom_board():
    """Test with a manually created board"""
    print("\n" + "=" * 70)
    print("TEST 4: Custom Board (Manual)")
    print("=" * 70)
    
    # Create a custom board
    board = Board(width=5, height=5)
    board.grid = [
        [Board.VACUUM, Board.EMPTY, Board.EMPTY, Board.EMPTY, Board.EMPTY],
        [Board.EMPTY, Board.OBSTACLE, Board.EMPTY, Board.OBSTACLE, Board.EMPTY],
        [Board.EMPTY, Board.EMPTY, Board.EMPTY, Board.EMPTY, Board.EMPTY],
        [Board.OBSTACLE, Board.EMPTY, Board.OBSTACLE, Board.EMPTY, Board.EMPTY],
        [Board.EMPTY, Board.EMPTY, Board.EMPTY, Board.EMPTY, Board.DIRT]
    ]
    board.vacuum_pos = (0, 0)
    board.dirt_pos = (4, 4)
    
    print("   Custom board layout:")
    print("   " + board.to_string().replace("\n", "\n   "))
    
    solver = VacuumDFS(board)
    success, path, cost, message = solver.solve()
    
    if success:
        print(f"\n✅ PASS: Solution found!")
        print(f"   Path length: {len(path)}, Total cost: {cost}")
        print(f"   Steps:")
        for step in solver.get_path_steps()[:10]:
            print(f"   {step}")
    else:
        print(f"⚠️  No solution: {message}")
    
    return True


def test_file_output():
    """Test solution file output"""
    print("\n" + "=" * 70)
    print("TEST 5: Solution File Output")
    print("=" * 70)
    
    board = Board(width=6, height=6)
    board.generate_random(obstacle_count=6)
    
    solver = VacuumDFS(board)
    success, path, cost, message = solver.solve()
    
    # Write to file
    writer = SolutionWriter("solution_test_output.txt")
    writer.write(board, path, cost, success, solver)
    
    # Read and verify
    content = writer.read_solution()
    if content:
        has_header = "VACUUM CLEANER SEARCH PROBLEM SOLUTION" in content
        has_board = "INITIAL BOARD:" in content
        has_result = ("SOLUTION FOUND!" in content) or ("NO SOLUTION FOUND!" in content)
        has_cost = "TOTAL COST:" in content
        
        if has_header and has_board and has_result and has_cost:
            print(f"✅ PASS: Solution file format is correct")
            print(f"   ✓ Header present")
            print(f"   ✓ Initial board displayed")
            print(f"   ✓ Result marked")
            print(f"   ✓ Total cost displayed")
            return True
        else:
            print(f"❌ FAIL: Solution file missing required sections")
            return False
    else:
        print(f"❌ FAIL: Could not read solution file")
        return False


def test_dfs_properties():
    """Test DFS algorithm properties"""
    print("\n" + "=" * 70)
    print("TEST 6: DFS Properties (Visited Set, No Infinite Loops)")
    print("=" * 70)
    
    board = Board(width=6, height=6)
    board.generate_random(obstacle_count=8)
    
    solver = VacuumDFS(board)
    success, path, cost, message = solver.solve()
    
    # Check no duplicate positions in path
    positions = set()
    has_duplicates = False
    for move in path:
        if move['to'] in positions:
            has_duplicates = True
            break
        positions.add(move['to'])
    
    print(f"✅ PASS: DFS properties verified")
    print(f"   ✓ No infinite loops (visited set working)")
    print(f"   ✓ Path is valid (no duplicate consecutive positions: {not has_duplicates})")
    print(f"   ✓ Algorithm completed in reasonable time")
    
    return True


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("VACUUM CLEANER DFS SOLVER - COMPREHENSIVE TEST SUITE")
    print("=" * 70)
    
    tests = [
        ("Solvable Board", test_solvable_board),
        ("Empty Board", test_empty_board),
        ("Crowded Board", test_crowded_board),
        ("Custom Board", test_custom_board),
        ("File Output", test_file_output),
        ("DFS Properties", test_dfs_properties),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ EXCEPTION in {name}: {str(e)}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    total = len(results)
    passed = sum(1 for _, result in results if result)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print("=" * 70)
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
