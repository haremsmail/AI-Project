# Week 3: A* Pathfinding Algorithm

An interactive GUI-based application that implements the **A* pathfinding algorithm** from scratch using Euclidean distance as the heuristic. Create nodes, draw edges, and visualize the optimal path finding process.

---

## 📋 Project Overview

This project implements a complete A* pathfinding algorithm with an interactive Tkinter GUI. Users can:
- Create custom nodes with coordinates
- Draw edges between nodes
- Visualize the graph on a grid-based board
- Run A* algorithm and see the optimal path
- View step-by-step exploration details
- Load a pre-configured sample graph for testing

### Key Features:
- 🎨 **Interactive GUI** with real-time visualization
- 🤖 **A* Algorithm** implemented from scratch (NO external libraries)
- 📊 **Euclidean Distance Heuristic** for optimal pathfinding
- 📁 **Node & Edge Management** with add/remove capabilities
- 🎯 **Step-by-Step Visualization** showing exploration order
- 🎲 **Sample Test Case** included for quick testing
- 💾 **Dynamic Node Labeling** (S, A, B, C, G, etc.)

---

## 🏗️ Project Structure

```
week3/
├── main.py                 # Entry point - launches the application
├── gui.py                  # GUI implementation using Tkinter
├── node.py                 # Node class - represents graph nodes
├── graph.py                # Graph class - manages nodes and edges
├── astar.py                # A* algorithm implementation (core logic)
├── requirements.txt        # Python dependencies
└── README.md               # This documentation
```

---

## 🧠 A* Algorithm Explanation

### Algorithm Overview

A* is a graph traversal algorithm that finds the shortest path between nodes by combining:
1. **g(n)**: Actual cost from start node to current node
2. **h(n)**: Estimated cost from current node to goal (Euclidean distance)
3. **f(n)**: Total estimated cost = g(n) + h(n)

### Key Concepts

**Heuristic (h)**: Uses Euclidean distance
```
h(n) = √((x₂ - x₁)² + (y₂ - y₁)²)
```

**Open List**: Priority queue of nodes to be explored (sorted by f cost)

**Closed List**: Set of already explored nodes

**Path Reconstruction**: Uses parent pointers to trace back the optimal path

### Algorithm Steps

1. Initialize start node with g=0
2. Add start node to open list
3. While open list is not empty:
   - Pop node with lowest f cost
   - If goal found, reconstruct and return path
   - Otherwise, expand neighbors:
     - Skip if already explored
     - Calculate new g cost
     - If better path found, update costs
     - Add to open list
4. Return empty if no path exists

---

## 🚀 How to Use

### Prerequisites
- Python 3.7+
- Tkinter (usually comes with Python)

### Installation

```bash
# Navigate to week3 directory
cd week3

# Install requirements (optional, tkinter is built-in)
pip install -r requirements.txt
```

### Running the Application

```bash
python main.py
```

---

## 🎯 User Guide

### 1. Creating Nodes

1. Enter node name (e.g., "S", "A", "G")
2. Enter X coordinate (0 to 10)
3. Enter Y coordinate (0 to 9)
4. Click "✚ ADD NODE"
5. Node appears on the grid

### 2. Creating Connections (Edges)

1. Left-click on a node to select it
2. Left-click on another node to create a directed edge
3. Edge is drawn from first node to second node

### 3. Removing Nodes/Edges

- Double-click a node to remove it (removes all connected edges)
- Right-click to deselect during edge creation

### 4. Running A* Algorithm

1. Select start node from dropdown
2. Select goal node from dropdown
3. Click "⚡ FIND PATH (A*)"
4. Path is highlighted in green
5. Explored nodes are highlighted in orange
6. Results show: path, cost, and nodes explored

### 5. Viewing Details

1. After running A*, click "📋 DETAILS"
2. Window shows:
   - Exploration order
   - g, h, f values for each step
   - Final path cost

### 6. Quick Testing

1. Click "📌 SAMPLE" to load pre-configured graph
2. Select start node "S" and goal node "G"
3. Click "⚡ FIND PATH (A*)"
4. See optimal path: S → A → C → G

---

## 📦 Sample Test Case

The application includes a pre-loaded sample graph:

### Nodes:
- **S** (Start) at (0, 0)
- **A** at (1, 2)
- **B** at (2, 1)
- **C** at (3, 3)
- **G** (Goal) at (5, 5)

### Edges (Connections):
```
S → A
S → B
A → C
B → C
C → G
```

### Expected Path:
**S → A → C → G** (or **S → B → C → G**)

Both paths have equal cost due to symmetric distances.

---

## 📊 Cost Calculation Example

For the sample graph, finding path S → G:

```
Step 1: Start at S (0,0)
  g(S) = 0
  h(S) = √((5-0)² + (5-0)²) = 7.07
  f(S) = 0 + 7.07 = 7.07

Step 2: Expand to A (1,2)
  g(A) = g(S) + dist(S,A) = 0 + √2 = 1.41
  h(A) = √((5-1)² + (5-2)²) = 5.00
  f(A) = 1.41 + 5.00 = 6.41

Step 3: From A, expand to C (3,3)
  g(C) = g(A) + dist(A,C) = 1.41 + √5 = 3.64
  h(C) = √((5-3)² + (5-3)²) = 2.83
  f(C) = 3.64 + 2.83 = 6.47

Step 4: From C, expand to G (5,5)
  g(G) = g(C) + dist(C,G) = 3.64 + √8 = 6.45
  h(G) = 0 (at goal)
  f(G) = 6.45 + 0 = 6.45

GOAL FOUND!
Path: S → A → C → G
Total Cost: 6.45
```

---

## 🎨 UI Color Scheme

- **Primary (Cyan)**: Titles and main elements
- **Success (Green)**: Final path, positive actions
- **Danger (Red)**: Goal node, errors
- **Warning (Orange)**: Explored nodes, statistics
- **Accent (Magenta)**: Special buttons
- **Node (Blue)**: Regular nodes
- **Grid (Dark Gray)**: Background grid

---

## 🔬 Code Architecture

### Node.py
- **Node Class**: Represents individual graph nodes
- Methods:
  - `euclidean_distance()`: Calculate distance to another node
  - `calculate_heuristic()`: Set h value based on goal
  - `update_f_cost()`: Calculate total f cost
  - `reset_costs()`: Prepare for new search

### Graph.py
- **Graph Class**: Manages all nodes and edges
- Methods:
  - `add_node()`: Add new node
  - `add_edge()`: Create connection between nodes
  - `get_node()`: Retrieve node by name
  - `is_connected()`: Check if path exists using BFS
  - `reset_all_costs()`: Reset for new search

### AStarFinder (astar.py)
- **A* Algorithm**: Core pathfinding implementation
- Methods:
  - `find_path()`: Main algorithm (uses open/closed lists)
  - `_reconstruct_path()`: Build path using parent pointers
  - `get_exploration_details()`: Return step-by-step info
  - `get_stats()`: Return search statistics

### AStarGUI (gui.py)
- **Tkinter GUI**: Interactive interface
- Methods:
  - `_add_node()`: Add node from user input
  - `_on_canvas_click()`: Handle node selection
  - `_solve()`: Run A* algorithm
  - `_redraw_canvas()`: Update visualization
  - `_load_sample_graph()`: Load test case

---

## 🧪 Testing

### Test Case 1: Sample Graph
- Load sample graph
- Start: S, Goal: G
- Expected path: S → A → C → G or S → B → C → G
- Expected cost: ~6.45

### Test Case 2: Single Path
- Nodes: S(0,0), A(5,0), G(10,0)
- Edges: S→A→G
- Expected path: S → A → G
- Expected cost: 10

### Test Case 3: Multiple Paths
- Create grid with alternative routes
- A* should choose path with lowest f cost
- Verify exploration order is correct

---

## 🎯 Key Algorithm Properties

1. **Admissibility**: A* finds optimal path when heuristic is admissible
   - h(n) ≤ actual cost to goal
   - Euclidean distance satisfies this for any metric

2. **Completeness**: A* will always find a path if one exists

3. **Optimality**: Path found has minimal cost (with admissible heuristic)

4. **Time Complexity**: O(b^d) where b=branching factor, d=depth
   - Depends on graph structure and heuristic quality

---

## 🐛 Troubleshooting

### No Path Found
- Check if nodes are actually connected
- Click "📋 DETAILS" to see explored nodes
- Use "📌 SAMPLE" to test with known working graph

### Wrong Path
- Verify node coordinates are correct
- Check edge directions (edges are directional)
- Reset and try again

### Performance Issues
- Reduce number of nodes/edges
- Check for cycles causing infinite loops
- Verify tree structure is not too deep

---

## 📝 Implementation Notes

- **No External Libraries**: A* implemented from scratch, only uses Tkinter for GUI
- **Priority Queue**: Uses Python's `heapq` module (standard library)
- **Euclidean Distance**: Calculated using math formula, not approximation
- **Parent Pointers**: Used for efficient path reconstruction
- **Grid-Based**: Nodes positioned on 10x9 grid with 50px per cell

---

## 🚀 Future Enhancements

- Bidirectional A* search
- Different heuristics (Manhattan, Chebyshev)
- Obstacle nodes (barriers to movement)
- Animated pathfinding visualization
- Export path to file
- Weight/cost input for edges
- Dijkstra's comparison
- Graph layout auto-arrangement

---

## 📄 License

This is an educational project for learning A* pathfinding algorithm.

---

## 👨‍💻 Author

Created as part of AI Algorithm Studies

---

## 📚 Resources Used

- A* Algorithm: https://en.wikipedia.org/wiki/A*_search_algorithm
- Euclidean Distance: Standard geometric formula
- Tkinter Documentation: Python GUI framework
- Heuristics for pathfinding: Admissibility principles

---

**Enjoy exploring the A* algorithm! 🚀**
