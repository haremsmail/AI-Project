# Vacuum Cleaner - DFS Solver

A complete implementation of the Vacuum Cleaner Search Problem using **Depth First Search (DFS)** algorithm with a modern GUI.

## 🎯 Problem Description

The vacuum cleaner needs to find a path from its current position to reach the dirt in a grid-based environment while avoiding obstacles.

**Grid Elements:**
- 🟦 **Vacuum (V)** - Blue, the agent
- 🟩 **Dirt (G)** - Green, the goal
- ⬛ **Obstacle (#)** - Black, cannot pass through
- ⬜ **Empty** - White, can move freely

## ⚙️ Algorithm Details

**Depth First Search (DFS)**
- Uses a stack-based approach (implicit via recursion)
- Explores as far as possible along each branch before backtracking
- Avoids infinite loops using a visited set

**Movement Costs:**
- **UP** → Cost: 2
- **DOWN** → Cost: 0
- **LEFT** → Cost: 1
- **RIGHT** → Cost: 1

## 📦 Project Structure

```
week2/
├── main.py                 # Entry point
├── gui.py                  # Modern Tkinter GUI
├── board.py               # Board generation & management
├── vacuum_dfs.py          # DFS algorithm implementation
├── solution_writer.py     # Solution file writer
├── solution.txt           # Output solution file (auto-generated)
└── README.md              # This file
```

## 🚀 Quick Start

### Requirements
- Python 3.7+
- tkinter (usually comes with Python)

### Installation

```bash
# Navigate to the week2 directory
cd week2

# No external dependencies needed! Just Python + tkinter
```

### Running the Application

```bash
python main.py
```

Or directly:

```bash
python gui.py
```

## 🎮 How to Use

1. **Generate Board** - Click "Generate New Board" to create a random 6×6 grid
2. **Solve** - Click "Solve with DFS" to find the path (runs in background)
3. **Show Path** - Click "Show Path" to animate the solution step-by-step
4. **View Results** - Check `solution.txt` for detailed output

## 📊 Features

### GUI Features
- ✅ Modern, clean interface
- ✅ Color-coded grid display
- ✅ Real-time status updates
- ✅ Step-by-step animation
- ✅ Solution path details
- ✅ Background solving (non-blocking)

### Algorithm Features
- ✅ DFS with visited set to avoid loops
- ✅ Cost calculation for each move
- ✅ Path tracking
- ✅ Backtracking when stuck
- ✅ No solution detection

## 💾 Output Files

### solution.txt Format
```
==================================================
VACUUM CLEANER SEARCH PROBLEM SOLUTION
Algorithm: Depth First Search (DFS)
Generated: 2026-03-28 12:34:56
==================================================

INITIAL BOARD:
V . . # . .
. . # . . .
. . . . . G
. # . . . .
. . . . # .
. . . . . .

Board Size: 6 x 6
Vacuum Position: (0, 0)
Dirt Position: (2, 5)

SOLUTION FOUND!
--------------------------------------------------

STEPS TO SOLUTION:
1. Move RIGHT from (0, 0) to (0, 1) (Cost: 1)
2. Move RIGHT from (0, 1) to (0, 2) (Cost: 1)
...

--------------------------------------------------
TOTAL COST: 15
TOTAL MOVES: 10
```

## 🎨 Code Design

### Clean Architecture
- **Modular design** - Each component has a single responsibility
- **Separation of concerns** - Algorithm, UI, board, and I/O are separate
- **Readable code** - Clear variable names and docstrings
- **Easy to extend** - Can be modified for other search algorithms

### Key Classes
```python
Board              # Grid management
VacuumDFS         # DFS algorithm
SolutionWriter    # File output
VacuumGUI         # UI interface
```

## 🔧 Customization

### Change Board Size
Edit `gui.py`, in `generate_board()`:
```python
self.board = Board(width=8, height=8)  # Default is 6x6
self.board.generate_random(obstacle_count=10)  # Adjust obstacles
```

### Change Animation Speed
Edit `gui.py`, in `_animate_step()`:
```python
self.root.after(800, self._animate_step)  # 800ms between steps (change this)
```

### Change Movement Costs
Edit `vacuum_dfs.py`:
```python
COSTS = {
    UP: 2,      # Change these values
    DOWN: 0,
    LEFT: 1,
    RIGHT: 1
}
```

## ✅ Testing

The GUI automatically:
1. Generates a random board
2. Solves using DFS
3. Writes results to `solution.txt`
4. Displays the solution with animation

Try multiple boards to see different scenarios:
- Easy paths (few obstacles)
- Hard paths (many obstacles)
- Impossible puzzles (blocked paths)

## 📝 Notes

- **No numpy needed** - Uses pure Python
- **No heavy dependencies** - Only tkinter (standard library)
- **Cross-platform** - Works on Windows, Mac, Linux
- **Efficient** - DFS explores efficiently with visited set
- **Memory-safe** - Proper backtracking prevents memory leaks

## 👨‍💻 Author

Created as an educational implementation of search algorithms with modern UI design.

---

**Enjoy solving vacuum cleaner puzzles! 🎉**
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
