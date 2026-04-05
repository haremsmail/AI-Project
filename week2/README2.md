# Week 2: Vacuum Cleaner AI Solver

A GUI-based AI project that uses **Depth-First Search (DFS)** algorithm to solve the vacuum cleaner problem. The project features an interactive board visualization, automated pathfinding, and solution documentation.

---

## 📋 Project Overview

This project implements a vacuum cleaner robot that must navigate a grid-based environment to reach dirt while avoiding obstacles. The solution uses the DFS algorithm to find an optimal path from the vacuum's starting position to the dirt location.

### Key Features:
- 🎨 **Interactive GUI** with real-time board visualization
- 🤖 **DFS Algorithm** for pathfinding
- 📊 **Cost Calculation** based on movement direction
- 📁 **Automatic Solution Export** to text file
- 🎯 **Step-by-Step Path Visualization**
- 🎲 **Random Board Generation** with customizable obstacles

---

## 🏗️ Project Structure

```
week2/
├── main.py                 # Entry point for the application
├── gui.py                  # GUI implementation using Tkinter
├── board.py                # Board/grid management
├── vacuum_dfs.py           # DFS pathfinding algorithm
├── solution_writer.py      # Solution file writer
├── solution.txt            # Generated solution output
├── requirements.txt        # Python dependencies
└── README2.md              # This file
```

---

## 📝 File Descriptions

### 1. **main.py**
- **Purpose**: Entry point for the application
- **Functionality**: Initializes and launches the GUI
- **Code**: Imports and calls `main()` from `gui.py`

### 2. **gui.py**
The main GUI component built with **Tkinter**. Features:

#### Key Classes:
- **VacuumGUI**: Main GUI window handler

#### Features:
- **Color Scheme**: Modern dark theme with neon accent colors
  - Background: Dark blue (#0a1628)
  - Primary accent: Cyan (#00d4ff)
  - Success: Green (#00ff88)
  - Danger: Red (#ff3366)
  
#### UI Components:
- **Canvas Board**: 420x420px grid visualization
- **Statistics Panel**:
  - Cost display
  - Status indicator (READY, SOLVING, FOUND, FAILED)
  - Moves counter
  
#### Buttons:
- **⟳ GENERATE**: Create random board with obstacles
- **▶ SOLVE**: Run DFS algorithm
- **⟫ STEP**: Move through solution step-by-step
- **💾 SAVE**: Export solution to file
- **🔄 RESET**: Clear board and restart
- **⚙ SETTINGS**: Configure obstacle count

#### Functionality:
- Real-time board visualization
- Background thread support for non-blocking solving
- Step-by-step visualization of solution path
- Dynamic color updates for board elements
- Solution logging to file

### 3. **board.py**
Grid/Board management system

#### Board Constants:
```python
EMPTY = 0        # Empty space
OBSTACLE = 1     # Obstacle (wall)
DIRT = 2         # Target (dirt to clean)
VACUUM = 3       # Robot/Vacuum cleaner
```

#### Symbol Mapping:
```
"." = Empty space
"#" = Obstacle
"G" = Dirt (Goal)
"V" = Vacuum (Vehicle)
```

#### Key Methods:
- **`__init__(width=6, height=6)`**: Initialize 6x6 grid
- **`generate_random(obstacle_count=6)`**: Generate random board with:
  - Random vacuum position
  - Random dirt position (different from vacuum)
  - Random obstacles (6 by default)
- **`is_valid_position(row, col)`**: Check if position is within bounds and not an obstacle
- **`get_symbol(row, col)`**: Get display symbol for a grid cell
- **`to_string()`**: Convert grid to string representation for display/file output

#### Properties:
- `grid`: 2D array representing the board
- `vacuum_pos`: Tuple (row, col) of vacuum position
- `dirt_pos`: Tuple (row, col) of dirt position
- `width`: Grid width (default 6)
- `height`: Grid height (default 6)

### 4. **vacuum_dfs.py**
DFS pathfinding algorithm implementation

#### Move Constants:
```python
UP = "UP"           # Upward movement
DOWN = "DOWN"       # Downward movement
LEFT = "LEFT"       # Leftward movement
RIGHT = "RIGHT"     # Rightward movement
```

#### Movement Costs:
```python
UP:    2 cost
DOWN:  0 cost
LEFT:  1 cost
RIGHT: 1 cost
```

#### Direction Vectors:
```python
UP:    (-1, 0)  # Decrease row
DOWN:  (1, 0)   # Increase row
LEFT:  (0, -1)  # Decrease column
RIGHT: (0, 1)   # Increase column
```

#### Key Classes:
- **Move**: Constants for movements and costs
- **VacuumDFS**: DFS solver

#### VacuumDFS Methods:
- **`__init__(board)`**: Initialize solver with board
- **`solve()`**: Main solving method
  - Returns: `(success: bool, path: list, cost: int, message: str)`
  - Uses DFS to find path from vacuum to dirt
  
- **`_dfs(current_pos)`**: Recursive DFS implementation
  - Marks visited positions
  - Tries all 4 directions (UP, DOWN, LEFT, RIGHT)
  - Backtracks if dead end is reached
  - Returns True when dirt is found
  
- **`get_path_steps()`**: Returns readable path steps
  - Format: `Move DIRECTION from (r1,c1) to (r2,c2) (Cost: X)`

#### Algorithm Logic:
1. Start at vacuum position
2. Mark current position as visited
3. Try moving in all 4 directions
4. Check if new position is valid and not visited
5. If dirt found, return success
6. If not found, backtrack and try other directions
7. Return path and total cost

### 5. **solution_writer.py**
Exports solutions to file

#### Key Methods:
- **`__init__(filename="solution.txt")`**: Initialize writer with output file
- **`write(board, path, total_cost, solution_found, solver)`**: Write complete solution with:
  - Header with timestamp
  - Initial board state
  - Board dimensions
  - Vacuum and dirt positions
  - Step-by-step solution (if found)
  - Total cost
  - Total number of moves
  - Error message (if not found)

#### Output Format:
- Section headers with formatting
- Visual board layout
- Detailed step list
- Solution statistics

---

## 🚀 How to Use

### Installation
```bash
# Install dependencies
pip install -r requirements.txt
```

### Running the Application
```bash
python main.py
```

### Using the GUI
1. **Generate Board**: Click "⟳ GENERATE" to create a random board
2. **View Board**: The canvas shows:
   - `V` = Vacuum (blue)
   - `G` = Dirt (green)
   - `#` = Obstacles (dark gray)
   - `.` = Empty spaces
3. **Solve**: Click "▶ SOLVE" to run the DFS algorithm
4. **View Solution**: Watch the status and cost update
5. **Step Through**: Use "⟫ STEP" button to advance one step at a time
6. **Save Solution**: Click "💾 SAVE" to export to `solution.txt`
7. **Reset**: Click "🔄 RESET" to clear and start over

---

## 📊 Algorithm Details

### DFS (Depth-First Search)
**Time Complexity**: O(V + E) where V = vertices, E = edges
**Space Complexity**: O(V) for visited set and recursion stack

### How DFS Works:
1. Start at vacuum position
2. Recursively explore adjacent unvisited cells
3. Mark each cell as visited to avoid cycles
4. When dirt is found, return success
5. If all directions fail, backtrack and try other paths

### Movement Cost System:
The cost is not distance-based but directional:
- Moving UP costs 2 units
- Moving DOWN costs 0 units
- Moving LEFT costs 1 unit
- Moving RIGHT costs 1 unit

This represents weighted directional preferences in the search space.

---

## 📝 Example Output

When you run SOLVE, the solution file (`solution.txt`) contains:

```
==================================================
VACUUM CLEANER SEARCH PROBLEM SOLUTION
Algorithm: Depth First Search (DFS)
Generated: 2026-04-05 10:30:45
==================================================

INITIAL BOARD:
V . . . . .
. # . . G .
. . . . . .
. # . . . .
. . . . . .
. . . . . .

Board Size: 6 x 6
Vacuum Position: (0, 0)
Dirt Position: (1, 4)

SOLUTION FOUND!
--------------------------------------------------

STEPS TO SOLUTION:
1. Move DOWN from (0, 0) to (1, 0) (Cost: 0)
2. Move RIGHT from (1, 0) to (1, 1) (Cost: 1)
3. Move RIGHT from (1, 1) to (1, 2) (Cost: 1)
4. Move RIGHT from (1, 2) to (1, 3) (Cost: 1)
5. Move RIGHT from (1, 3) to (1, 4) (Cost: 1)

--------------------------------------------------
TOTAL COST: 4
TOTAL MOVES: 5
```

---

## ⚙️ Configuration

### Obstacle Count
Edit in GUI settings or directly in code:
```python
self.board.generate_random(obstacle_count=6)  # Default: 6
```

### Board Size
Modify in `board.py`:
```python
Board(width=6, height=6)  # Default: 6x6
```

### Color Theme
Customize in `gui.py` COLORS dictionary:
```python
COLORS = {
    'bg': '#0a1628',        # Background
    'primary': '#00d4ff',   # Default accent
    'success': '#00ff88',   # Success color
    # ... more colors
}
```

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| GUI window not appearing | Ensure tkinter is installed: `pip install tk` |
| No solution found | Increase board size or reduce obstacles |
| Solution file not created | Ensure write permissions in directory |
| Algorithm running slow | Reduce board size or obstacle count |

---

## 📚 Dependencies

- **tkinter**: GUI framework (included with Python)
- **threading**: Background task execution
- **random**: Random board generation
- **datetime**: Solution timestamp logging

See `requirements.txt` for complete list.

---

## 🎓 Learning Outcomes

This project demonstrates:
- ✅ Implementing DFS algorithm from scratch
- ✅ GUI development with Tkinter
- ✅ Grid-based pathfinding problems
- ✅ State space search
- ✅ Backtracking algorithms
- ✅ File I/O operations
- ✅ Multi-threading in Python
- ✅ Dynamic UI updates

---

## 📄 License

This is an educational project for Week 2 of the AI-Project course.

---

## 📞 Notes

- The algorithm always finds a solution unless obstacles completely block the path
- The solution may not be the shortest path, but rather the first path found by DFS
- Cost calculation follows the directional system defined in `Move.COSTS`
- Solutions are saved with timestamps to prevent data loss

---

**Last Updated**: April 5, 2026  
**Author**: AI-Project Team  
**Project Status**: Complete and Functional
