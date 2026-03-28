# Vacuum Cleaner DFS Solver - Quick Start Guide

## 🚀 Get Started in 30 Seconds

### Step 1: Open Terminal
```bash
cd c:\Desktop\week1\AI-Project\week2
```

### Step 2: Run the Application
```bash
python main.py
```

Or double-click `run.bat` on Windows.

### Step 3: Use the GUI
1. Click **"Generate New Board"** to create a random puzzle
2. Click **"Solve with DFS"** to find the solution
3. Click **"Show Path"** to see the animation
4. Check `solution.txt` for detailed output

---

## 📊 What You'll See

```
┌─────────────────────────────────────────┐
│  Vacuum Cleaner - DFS Solver            │
├─────────────────────────────────────────┤
│ [Generate] [Solve] [Show Path]          │
├─────────────────────────────────────────┤
│                   │  Status: Ready      │
│   Board Display   │  Cost: -            │
│   (6x6 Grid)      │  Moves: -           │
│   Colored Cells   │                     │
│                   │  Legend:            │
│                   │  🟦 Vacuum          │
│                   │  🟩 Dirt            │
│                   │  ⬛ Obstacle        │
│                   │  ⬜ Empty           │
│                   │                     │
│                   │  Solution steps...  │
└─────────────────────────────────────────┘
```

---

## 🎮 Button Guide

| Button | Action | When Available |
|--------|--------|-----------------|
| **Generate New Board** | Creates a new random 6×6 grid | Always |
| **Solve with DFS** | Finds the path using DFS algorithm | After generating board |
| **Show Path** | Animates the solution step-by-step | After solving |

---

## 📁 Files Explained

| File | Purpose |
|------|---------|
| `main.py` | Start the application here |
| `gui.py` | The graphical interface |
| `board.py` | Manages the grid and obstacles |
| `vacuum_dfs.py` | The DFS algorithm |
| `solution_writer.py` | Writes results to file |
| `solution.txt` | Auto-generated solution output |
| `run.bat` | Windows shortcut to run the app |

---

## 🎨 Visual Guide

### Color Meanings
```
🟦 Blue (V)   = Vacuum (your robot)
🟩 Green (G)  = Dirt (goal/target)
⬛ Black (#)  = Obstacle (cannot pass)
⬜ White (.)  = Empty space
🟧 Orange    = Path taken (during animation)
```

### Example Board
```
Initial:              Solution:
V . # . . .          V→→→. . .
. . . . . .          . . . . . .
. . . . . .          . . . . . .
. . . . . .   =>     . . . . . .
. . . . . .          . . . . . .
# . . G . .          # . . G . .

Path: RIGHT, RIGHT, RIGHT, DOWN, DOWN, DOWN, DOWN, DOWN, LEFT
Cost: 1 + 1 + 1 + 0 + 0 + 0 + 0 + 0 + 1 = 4
```

---

## ✅ What the Program Does

### Step 1: Generate Board (Automatic)
- Creates a random 6×6 grid
- Randomly places vacuum (blue)
- Randomly places dirt (green)
- Randomly places 8 obstacles (black)
- Shows the board on screen

### Step 2: Solve (Click "Solve with DFS")
- Runs DFS algorithm in background
- Searches for path from vacuum to dirt
- Avoids obstacles and visited cells
- Calculates total cost
- Displays results in "Solution steps" box

### Step 3: Show Path (Click "Show Path")
- Animates the solution
- One step every 800 milliseconds
- Highlights the path being taken
- Updates status in real-time
- Shows final statistics

### Step 4: Check Results
- Open `solution.txt` to see:
  - Initial board state
  - Step-by-step moves
  - Total cost calculation
  - Algorithm details

---

## 📊 Example Output File

```
==================================================
VACUUM CLEANER SEARCH PROBLEM SOLUTION
Algorithm: Depth First Search (DFS)
Generated: 2026-03-28 12:34:56
==================================================

INITIAL BOARD:
V . . # . .
. . . . . .
. . . . . G
# . . . . .
. . . . . .
. . . . . .

Board Size: 6 x 6
Vacuum Position: (0, 0)
Dirt Position: (2, 5)

SOLUTION FOUND!
--------------------------------------------------

STEPS TO SOLUTION:
1. Move RIGHT from (0, 0) to (0, 1) (Cost: 1)
2. Move RIGHT from (0, 1) to (0, 2) (Cost: 1)
3. Move DOWN from (0, 2) to (1, 2) (Cost: 0)
4. Move DOWN from (1, 2) to (2, 2) (Cost: 0)
5. Move RIGHT from (2, 2) to (2, 3) (Cost: 1)
6. Move RIGHT from (2, 3) to (2, 4) (Cost: 1)
7. Move RIGHT from (2, 4) to (2, 5) (Cost: 1)

--------------------------------------------------
TOTAL COST: 5
TOTAL MOVES: 7

==================================================
```

---

## 🔧 Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'tkinter'"
**Solution:** Install tkinter
```bash
# Windows
pip install tk

# Mac
brew install python-tk@3.9

# Linux (Ubuntu/Debian)
sudo apt install python3-tk
```

### Problem: GUI doesn't appear
**Solution:** Ensure tkinter is installed and try:
```bash
python -c "import tkinter; print('OK')"
```

### Problem: "No solution because of obstacles"
**Solution:** This is normal! Some randomly generated boards have no path. Just click "Generate New Board" to try again.

### Problem: Can't find solution.txt
**Solution:** Check the week2 folder. It's created when you click "Solve".

---

## 📚 Understanding DFS Algorithm

### How DFS Works

1. **Start** at vacuum position
2. **Explore** each direction (UP, DOWN, LEFT, RIGHT)
3. **Mark** visited cells to avoid loops
4. **Go deeper** until you hit a dead end (backtrack)
5. **Try other routes** until you find dirt
6. **Calculate cost** based on movement directions

### Movement Costs
```
UP    → costs 2 points
DOWN  → costs 0 points (easy!)
LEFT  → costs 1 point
RIGHT → costs 1 point
```

### Why DFS?
✅ Memory efficient
✅ Finds solution quickly
✅ Easy to understand
✅ Works great for small grids

---

## 🎓 Educational Value

This project teaches:
- **Algorithm Design:** How DFS actually works
- **Data Structures:** Visited set, Path tracking
- **GUI Programming:** Tkinter, Threading
- **Software Design:** Clean architecture, separation of concerns
- **Testing:** Comprehensive test strategies

---

## 💡 Try These Experiments

### Experiment 1: Different Starting Positions
Edit `board.py` to place vacuum at different starting positions and observe solution changes.

### Experiment 2: Varying Obstacle Counts
In GUI `generate_board()`, change `obstacle_count` from 5 to 20 and observe difficulty.

### Experiment 3: Different Movement Costs
In `vacuum_dfs.py`, change costs to `UP=1, DOWN=1, LEFT=1, RIGHT=1` and see path optimization.

### Experiment 4: Larger Boards
Change board size from 6×6 to 8×8 or 10×10 and measure solve time.

---

## 🎯 Next Steps

1. **Run the application:** `python main.py`
2. **Generate a board:** Click "Generate New Board"
3. **Solve it:** Click "Solve with DFS"
4. **See the path:** Click "Show Path"
5. **Check output:** Look at `solution.txt`
6. **Repeat:** Generate different boards and solve them!

---

## 📞 Getting Help

- **Code questions?** Check `CODE_DOCUMENTATION.md`
- **Algorithm questions?** Look at comments in `vacuum_dfs.py`
- **GUI questions?** See `gui.py` for UI implementation
- **Test the code?** Run `python test_dfs.py` or `python test_comprehensive.py`

---

## 🎉 Enjoy!

You now have a working Vacuum Cleaner DFS Solver with a beautiful GUI!

**Key Features You've Got:**
✅ Modern GUI interface
✅ Working DFS algorithm
✅ Random board generation
✅ Step-by-step animation
✅ Solution file output
✅ Comprehensive tests

**Have fun exploring the vacuum cleaner world! 🤖**
