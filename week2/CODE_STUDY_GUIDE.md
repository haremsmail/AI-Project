# 📚 Code Study Guide - Vacuum Cleaner DFS Solver

## **START HERE! 🎯**

This guide shows you **which file to read first** and **how they connect** to each other.

---

## **Step 1️⃣: Entry Point - START WITH main.py**

**File:** `main.py`  
**What it does:** This is the FIRST file to run - it's the entry point  
**Size:** Very small (14 lines)

```python
from gui import main

if __name__ == "__main__":
    main()
```

**What happens:**
1. Imports the GUI module
2. Calls `main()` function from `gui.py`
3. Your application starts!

**After reading main.py → GO TO: gui.py** ✅

---

## **Step 2️⃣: GUI Interface - THEN READ gui.py**

**File:** `gui.py`  
**What it does:** Creates the graphical interface (buttons, board display, controls)  
**Size:** Large (850+ lines)

### **Important Classes/Methods in gui.py:**

| Method | What it does |
|--------|-------------|
| `__init__()` | Sets up the GUI and initializes variables |
| `setup_ui()` | Creates the layout (title, board, buttons, panel) |
| `generate_board()` | Creates a new random board |
| `solve()` | Starts solving in background thread |
| `_solve_background()` | Runs DFS algorithm (calls VacuumDFS) |
| `on_canvas_click()` | Handles manual play mode |
| `draw_board()` | Displays the board on screen |
| `next_step()` / `previous_step()` | Navigate through solution steps |
| `reset_solution()` | Resets back to start |

### **Key Flow in gui.py:**
```
User clicks GENERATE
    ↓
generate_board() → creates Board object
    ↓
User clicks SOLVE
    ↓
solve() → starts background thread
    ↓
_solve_background() → calls VacuumDFS.solve()
    ↓
Solution received → draw_board() displays it
```

**After reading gui.py → GO TO: board.py** ✅

---

## **Step 3️⃣: Board Logic - THEN READ board.py**

**File:** `board.py`  
**What it does:** Represents the 6×6 grid with vacuum, dirt, and obstacles  
**Size:** Small (120 lines)

### **Important Classes in board.py:**

```python
class Board:
    # Constants for grid values
    EMPTY = 0      # Empty cell
    OBSTACLE = 1   # Wall/obstacle
    VACUUM = 2     # Robot position
    DIRT = 3       # Target/goal
    
    # Methods:
    generate_random()      # Creates random board
    is_valid_position()    # Checks if position has no obstacle
    get_neighbors()        # Gets adjacent cells
```

### **Board Grid Example:**
```
0 = Empty (white)
1 = Obstacle (dark)
2 = Vacuum (blue V)
3 = Dirt/Goal (green G)

Example:
[0, 0, 0, 0, 1, 2]  ← Vacuum at position (0,5)
[0, 1, 1, 0, 0, 0]
[0, 0, 0, 1, 0, 0]
[1, 0, 1, 0, 0, 1]
[0, 0, 0, 0, 3, 0]  ← Dirt/Goal at position (4,4)
[0, 1, 0, 0, 1, 0]
```

**After reading board.py → GO TO: vacuum_dfs.py** ✅

---

## **Step 4️⃣: DFS Algorithm - THEN READ vacuum_dfs.py**

**File:** `vacuum_dfs.py`  
**What it does:** Implements Depth-First-Search algorithm to find path from vacuum to dirt  
**Size:** Medium (180 lines)

### **Important Methods:**

```python
class VacuumDFS:
    solve()              # Main algorithm - finds path from start to goal
    _dfs()               # Recursive DFS helper function
    _get_move_cost()     # Calculates cost (UP=2, DOWN=0, LEFT=1, RIGHT=1)
```

### **How DFS Works:**
```
Start at vacuum position
    ↓
Explore all possible moves (UP, DOWN, LEFT, RIGHT)
    ↓
For each move:
  - Check if valid (no obstacle, within bounds)
  - Check if we reached dirt (goal)
  - If not, explore deeper (recursive)
    ↓
When goal found → Return the path
When dead end → Backtrack and try other moves
```

### **Return Value:**
```python
success, path, cost, message = solver.solve()

# success = True/False (found solution or not)
# path = list of moves like:
#   [{'from': (0,0), 'to': (1,0), 'move': 'DOWN', 'cost': 0}, ...]
# cost = total cost of all moves
# message = description of result
```

**After reading vacuum_dfs.py → GO TO: solution_writer.py** ✅

---

## **Step 5️⃣: File Output - THEN READ solution_writer.py**

**File:** `solution_writer.py`  
**What it does:** Writes the solution to `solution.txt` file in a nice format  
**Size:** Small (80 lines)

### **Important Methods:**

```python
class SolutionWriter:
    write()          # Writes solution to solution.txt
    read_solution()  # Reads and returns content from solution.txt
```

### **What Gets Written to solution.txt:**
```
INITIAL BOARD:
[0, 0, 1, 0]
[2, 0, 0, 3]
...

STEPS TO SOLUTION:
1. DOWN (Cost: 0) - Total: 0
2. RIGHT (Cost: 1) - Total: 1
3. RIGHT (Cost: 1) - Total: 2
...

FINAL COST: 5
STATUS: GOAL REACHED!
```

**After reading solution_writer.py → YOU'RE DONE! 🎉** ✅

---

## **Complete File Flow Diagram:**

```
main.py (entry point)
    ↓
gui.py (create GUI)
    ├─ calls → board.py (creates board)
    ├─ calls → vacuum_dfs.py (solves puzzle)
    └─ calls → solution_writer.py (saves results)
    
User Interactions:
├─ GENERATE → board.generate_random()
├─ SOLVE → vacuum_dfs.solve()
├─ NEXT/PREVIOUS → navigate through path
├─ PLAY MANUALLY → on_canvas_click()
└─ solution.txt is auto-created by solution_writer.write()
```

---

## **Study Tips 💡:**

1. **Read files in order:** main → gui → board → vacuum_dfs → solution_writer
2. **Keep a notepad:** Write down what each function does
3. **Trace an example:** 
   - Generate a board
   - Click SOLVE
   - Watch what happens in each file
4. **Test manual mode:** Click PLAY to understand the clicking mechanism
5. **Check solution.txt:** After solving, read the output file

---

## **Key Concepts to Understand:**

| Concept | Where to learn | What it means |
|---------|----------------|---------------|
| **Board** | board.py | The 6×6 grid with vacuum, dirt, obstacles |
| **DFS** | vacuum_dfs.py | Algorithm that explores all paths |
| **Cost** | vacuum_dfs.py | UP=2, DOWN=0, LEFT=1, RIGHT=1 |
| **GUI** | gui.py | User interface and visualization |
| **Solution Path** | solution_writer.py | List of moves from start to goal |

---

## **Example: Complete Execution Flow**

```
1. User runs: python main.py
   └─ main.py imports gui.VacuumGUI

2. User clicks GENERATE
   └─ gui.generate_board() 
      └─ board.Board.generate_random()
         └─ Creates 6x6 grid with random obstacles
         └─ Places vacuum (V) and dirt (G)

3. User clicks SOLVE
   └─ gui.solve()
      └─ Starts background thread: _solve_background()
         └─ Creates vacuum_dfs.VacuumDFS object
         └─ Calls vacuum_dfs.solve()
            └─ Uses DFS algorithm to find path
            └─ Returns: success, path, cost, message

4. Solution received
   └─ solution_writer.write() saves to solution.txt
   └─ gui.draw_board() displays path visually
   └─ User can click NEXT to see each step

5. User clicks NEXT
   └─ gui.next_step()
      └─ draw_board(highlight_step=current_step)
         └─ Shows current step with orange highlight
```

---

## **Questions to Ask Yourself While Reading:**

- [ ] **main.py**: What does it import?
- [ ] **gui.py**: What happens when I click GENERATE?
- [ ] **gui.py**: What happens when I click SOLVE?
- [ ] **board.py**: How is the board represented? (0, 1, 2, 3 values)
- [ ] **board.py**: What is `is_valid_position()`?
- [ ] **vacuum_dfs.py**: What is DFS and how does it work?
- [ ] **vacuum_dfs.py**: How are costs calculated?
- [ ] **solution_writer.py**: What format is the solution.txt file?
- [ ] **All**: How do all files work together?

---

## **🎯 Your Study Checklist**

- [ ] Read main.py (5 min)
- [ ] Read gui.py - understand setup_ui() (20 min)
- [ ] Read gui.py - understand solve() flow (15 min)
- [ ] Read board.py - understand grid structure (10 min)
- [ ] Read vacuum_dfs.py - understand DFS algorithm (20 min)
- [ ] Read solution_writer.py - understand file output (5 min)
- [ ] Run the program and test all features (10 min)
- [ ] Try manual play mode (5 min)
- [ ] Check solution.txt output file (5 min)

**Total study time: ~95 minutes** ⏱️

Good luck! Happy learning! 📚✨
