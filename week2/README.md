# Search Algorithms Solver - Week 2 Project

A modern GUI application that solves both 8-Puzzle and Vacuum World Problems using three different search algorithms:
- Depth-First Search (DFS)
- Breadth-First Search (BFS)
- Best-First Search (heuristic guided)

## Features

### 8-Puzzle Solver
- Solve random 8-puzzle boards
- Choose between DFS, BFS, or Best-FS (with Manhattan Distance heuristic)
- Visual step-by-step solution display
- Solution saved to solution.txt
- Track total steps and cost

### Vacuum World Solver
- Random world generation with obstacles and dirt
- Three different search algorithms
- Variable move costs (Top: 2, Bottom: 0, Left/Right: 1)
- Visual representation of walls, dirt, and vacuum
- Handles impossible scenarios (blocked by obstacles)
- Solution saved to solution.txt

## Installation

No external dependencies required beyond Python standard library and tkinter (included with Python).

```bash
python main.py
```

## Project Structure

```
week2/
├── main.py                 # GUI Application
├── search_algorithms.py    # DFS, BFS, Best-FS implementations
├── puzzle_solver.py        # 8-Puzzle specific logic
├── vacuum_solver.py        # Vacuum World specific logic
└── solution.txt            # Latest generated solution report
```

## Usage

1. Run `python main.py`
2. Choose your problem type tab (8-Puzzle or Vacuum World)
3. Select one algorithm: DFS, BFS, or Best-FS
4. Click "Solve" (computation runs in a background thread)
5. Use Next/Previous buttons to navigate all steps visually
6. **Solution is automatically saved to `solution.txt`** after solving
   - If you click "Save Solution" button again, it updates `solution.txt` with current result
   - The old content is replaced with the new solution
   - Each new solve will update the file when you click Solve

## Required Tasks (A-F)

### A. 8-puzzle DFS
1. Open tab: `8-Puzzle Solver`
2. Select `Depth-First Search`
3. Click `Generate New Puzzle`
4. Click `Solve`
5. Navigate with `Next Step` and `Previous Step`
6. Check `solution.txt` for initial board, all steps, and overall cost

### B. 8-puzzle BFS
1. Open tab: `8-Puzzle Solver`
2. Select `Breadth-First Search`
3. Click `Generate New Puzzle`
4. Click `Solve`
5. Navigate steps on the board
6. Check `solution.txt`

### C. 8-puzzle Best-FS
1. Open tab: `8-Puzzle Solver`
2. Select `Best-First Search (A*)`
3. Click `Generate New Puzzle`
4. Click `Solve`
5. Navigate steps on the board
6. Check `solution.txt`

### D. Vacuum DFS
1. Open tab: `Vacuum World Solver`
2. Select `Depth-First Search`
3. Click `Generate New World`
4. Click `Solve`
5. Navigate steps on the board
6. Check `solution.txt` for board, steps, and total cost (or no-solution message)

### E. Vacuum BFS
1. Open tab: `Vacuum World Solver`
2. Select `Breadth-First Search`
3. Click `Generate New World`
4. Click `Solve`
5. Navigate steps on the board
6. Check `solution.txt`

### F. Vacuum Best-FS
1. Open tab: `Vacuum World Solver`
2. Select `Best-First Search`
3. Click `Generate New World`
4. Click `Solve`
5. Navigate steps on the board
6. Check `solution.txt`

## Algorithms

### Depth-First Search (DFS)
- Explores as far as possible along each branch
- Uses a stack-based approach
- Less memory efficient for 8-puzzle

### Breadth-First Search (BFS)
- Explores all neighbors at depth k before moving to k+1
- Uses a queue-based approach
- Guarantees shortest path (by number of steps)

### Best-First Search
- Uses heuristics to guide search
- 8-Puzzle: Manhattan Distance heuristic
- Vacuum: Manhattan Distance to target
- Most efficient for both problems

## Solution Files

Generated solution files contain:
- Initial board state
- Step-by-step progression
- Cost of each move
- Total cost and steps
- Algorithm used

## Author

AI-Project Group
