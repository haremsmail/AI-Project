# PSO Visualization Project - Complete Analysis

## ✅ Assignment Requirements Verification

### ✓ Requirement 1: Main Logic in Python
**Status:** ✅ COMPLETE
- **File:** `pso_logic.py` (160+ lines)
- **Components:**
  - `Particle` class - represents individual particles
  - `run_pso_step()` function - executes PSO algorithm
  - All PSO mathematics implemented manually from scratch
  - No external AI/optimization libraries used

### ✓ Requirement 2: Randomly Place Particles
**Status:** ✅ COMPLETE
- **File:** `ui.py` - `start()` method
- **Implementation:**
  - N particles (user-defined, default 10)
  - Spawned at cluster center on opposite side of canvas from goal
  - Random offset ±40 units from cluster center for natural distribution
  - Each particle gets unique ID and color for identification

```python
# Cluster placement logic (ui.py, line 91-92)
cx = 50 if gx > 310 else 550    # Opposite side horizontally
cy = 50 if gy > 325 else 600    # Opposite side vertically
```

### ✓ Requirement 3: Goal Position and Parameters by User
**Status:** ✅ COMPLETE
- **GUI Input Fields:** `PSO Parameters` panel
- **User-Configurable Inputs:**
  - **Particles (n):** Number of particles (default: 10)
  - **Goal X:** X-coordinate of goal (default: 310)
  - **Goal Y:** Y-coordinate of goal (default: 400)
  - **w:** Inertia weight (default: 0.7)
  - **c1:** Cognitive coefficient (default: 1.5)
  - **c2:** Social coefficient (default: 1.5)
  - **Max Iter:** Maximum iterations (default: 100)

### ✓ Requirement 4: Goal in Different Color
**Status:** ✅ COMPLETE
- **File:** `ui.py` - `_draw()` method
- **Implementation:**
  - Goal displayed as pink circle: `#f38ba8`
  - Star symbol (★) inside for visibility
  - Size: 12-pixel radius
  - Always visible and distinguishable from particles

```python
# Goal rendering (ui.py, line 146-148)
self.canvas.create_oval(goal_x - r, goal_y - r, goal_x + r, goal_y + r,
                       fill="#f38ba8", outline="#fab387", width=2)
self.canvas.create_text(goal_x, goal_y, text="★", fill="#1e1e2e", font=("Arial", 12, "bold"))
```

### ✓ Requirement 5: Animation Speed Control
**Status:** ✅ COMPLETE
- **File:** `ui.py` - `_animate()` method
- **Control:** Speed slider widget
- **Range:** 1-200 milliseconds (1ms = fastest, 200ms = slowest)
- **Implementation:**
  - `tk.Scale` slider for visual adjustment
  - Real-time speed adjustment during animation
  - Used in: `self.root.after(self.speed_var.get(), lambda: self._animate(params))`

### ✓ Requirement 6: Save Iteration History & Selection Ability
**Status:** ✅ COMPLETE
- **File:** `ui.py` - `_animate()` method
- **History Storage:**
  - Each iteration saved with:
    - Iteration number
    - Best position (best_x, best_y)
    - Fitness value
    - All particle positions
  - Stored in `self.history` list (Python dictionary)

```python
# History recording (ui.py, line 124-127)
self.history.append({
    "iter": self.iteration,
    "best_x": self.global_best[0],
    "best_y": self.global_best[1],
    "fitness": best_fit,
    "positions": snapshot
})
```

- **Selection UI:**
  - Listbox widget showing all iterations
  - Format: `Iter XXX | Fit: XXXX.XX`
  - Click to select, then click "📌 Show Selected" button
  - Scrollable history for large iteration counts

### ✓ Requirement 7: Particle Differentiation
**Status:** ✅ COMPLETE
- **File:** `ui.py` - COLORS array + `_draw()` method
- **Differentiation Methods:**
  1. **Color:** 20 unique colors from Catppuccin palette
  2. **Numbering:** Each particle numbered 1-20 displayed on canvas
  - Particles drawn as circles with distinct visual identity
  - Colors cycle through palette if more than 20 particles

```python
# Color palette (ui.py, line 7-9)
COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4", 
          "#f032e6", "#bfef45", "#fabed4", "#469990", "#dcbeff", "#9A6324", 
          "#800000", "#aaffc3", "#808000", "#000075", "#a9a9a9", "#e6beff", "#ffe119", "#ffd8b1"]
```

### ✓ Requirement 8: No External PSO Libraries
**Status:** ✅ COMPLETE
- **PSO Formula Implemented Manually:**
  ```
  v_new = w*v_old + c1*r1*(pBest - x) + c2*r2*(gBest - x)
  x_new = x + v_new
  ```
- **File:** `pso_logic.py`
- **Methods:**
  - `Particle.update_velocity()` - PSO velocity formula
  - `Particle.update_position()` - Position update
  - `Particle.update_personal_best()` - Personal best tracking
  - `run_pso_step()` - Complete PSO iteration

---

## 📋 Project Architecture

### File Structure
```
week4/
├── main.py              (30 lines) - Entry point
├── pso_logic.py         (160 lines) - PSO algorithm
├── ui.py                (175 lines) - GUI visualization
├── requirements.txt     (3 lines) - Dependencies
└── PROJECT_ANALYSIS.md  (this file)
```

### Component Diagram
```
┌─────────────────────────────────────────────────────────┐
│                      main.py                            │
│              (Application Entry Point)                  │
└──────────────────────────┬──────────────────────────────┘
                           │
                    Initializes
                           │
        ┌──────────────────┴──────────────────┐
        │                                     │
        ▼                                     ▼
   ┌─────────────┐                   ┌─────────────────┐
   │  pso_logic  │                   │      ui.py      │
   ├─────────────┤                   ├─────────────────┤
   │ • Particle  │◄──────────────────┤ • PSOApp class  │
   │ • run_pso   │   Uses algorithm  │ • GUI rendering │
   │   _step()   │                   │ • User controls │
   └─────────────┘                   └─────────────────┘
        ▲                                     ▲
        │                                     │
        └─────────────────┬────────────────────┘
                      Data Flow
```

### Class Hierarchy

#### Particle Class (pso_logic.py)
```python
class Particle:
    - x, y                 # Current position
    - vx, vy               # Velocity components
    - best_x, best_y       # Personal best position
    - best_fitness         # Personal best fitness
    - color                # Unique color for display
    
    Methods:
    - fitness()            # Calculate distance to goal
    - update_personal_best()  # Update pBest
    - update_velocity()    # Apply PSO formula
    - update_position()    # Move particle
    - get_speed()          # Return velocity magnitude
```

#### PSOApp Class (ui.py)
```python
class PSOApp:
    - particles            # List of Particle objects
    - running              # Animation state
    - iteration            # Current iteration number
    - history              # List of saved states
    - global_best          # Best position found
    
    Methods:
    - _build_ui()          # Create GUI components
    - _get_params()        # Read user inputs
    - start()              # Initialize and begin
    - pause()              # Pause animation
    - reset()              # Clear state
    - _animate()           # Main loop
    - _draw()              # Render visualization
    - show_selected()      # Display selected iteration
```

---

## 🔄 Program Flow

### 1. Application Start (main.py)
```
main.py
  ↓
Create Tkinter root window
  ↓
Initialize PSOApp(root)
  ↓
Display GUI
  ↓
Wait for user interaction
```

### 2. User Clicks "Start" Button
```
start() method called
  ↓
Read parameters (n, gx, gy, w, c1, c2, max_iter)
  ↓
Validate inputs (try/except)
  ↓
Create N particles at cluster center
  ↓
Initialize global_best = (goal_x, goal_y, infinity)
  ↓
Set running = True
  ↓
Call _animate()
```

### 3. Animation Loop (per frame)
```
_animate() called
  ↓
Check if max_iter reached → stop if yes
  ↓
Call run_pso_step():
  ├─ For each particle:
  │  ├─ Calculate fitness to goal
  │  ├─ Update personal best if better
  │  └─ Update global best if better
  │
  ├─ For each particle:
  │  ├─ Update velocity using PSO formula
  │  └─ Update position (move particle)
  │
  └─ Return new global_best
  ↓
Increment iteration counter
  ↓
Save snapshot to history
  ↓
Update display (info label)
  ↓
Render canvas (_draw())
  ↓
Schedule next frame: self.root.after(speed_ms, _animate)
```

### 4. Canvas Rendering
```
_draw() called
  ↓
Clear canvas
  ↓
Draw grid background
  ↓
Draw goal (pink star)
  ↓
Draw best-found position (green cross)
  ↓
For each particle:
  ├─ Draw colored circle
  ├─ Draw particle number
  └─ Update screen
```

### 5. User Selects History Item
```
User clicks iteration in Listbox
  ↓
User clicks "📌 Show Selected" button
  ↓
show_selected() retrieves record from history
  ↓
Clear canvas and redraw:
  ├─ Grid
  ├─ Goal position
  ├─ Particle positions from snapshot
  └─ Best position from that iteration
```

---

## 📊 PSO Algorithm Details

### Fitness Function
```python
fitness(goal_x, goal_y) = √[(x - goal_x)² + (y - goal_y)²]
```
- **Meaning:** Euclidean distance from particle to goal
- **Goal:** Minimize fitness (lower = closer to goal)
- **Convergence:** When fitness approaches 0

### PSO Update Formula
```python
v_new = w*v_old + c1*r1*(pBest - x) + c2*r2*(gBest - x)
x_new = x + v_new
```

**Components:**
1. **Inertia term (w*v_old):** 
   - Keeps particle moving in same direction
   - w=0.7 gives moderate momentum
   
2. **Cognitive term (c1*r1*(pBest-x)):**
   - Pulls particle toward its own best position
   - c1=1.5 is standard coefficient
   - r1 = random [0,1] for stochasticity
   
3. **Social term (c2*r2*(gBest-x)):**
   - Pulls particle toward global best
   - c2=1.5 equals cognitive term
   - Balances exploration vs exploitation

### Pseudo Code Implementation
```python
for each iteration:
    for each particle:
        fitness = distance(particle, goal)
        if fitness < personal_best:
            personal_best = fitness
        
        if fitness < global_best:
            global_best = fitness
    
    for each particle:
        r1, r2 = random(), random()
        vx = w*vx + c1*r1*(pbest_x-x) + c2*r2*(gbest_x-x)
        vy = w*vy + c1*r1*(pbest_y-y) + c2*r2*(gbest_y-y)
        x += vx
        y += vy
```

---

## 🎮 User Interface Guide

### Control Panel (Right Side)
| Control | Type | Range | Purpose |
|---------|------|-------|---------|
| Particles | Input | 1-20 | Number of particles |
| Goal X | Input | 0-620 | Goal X coordinate |
| Goal Y | Input | 0-650 | Goal Y coordinate |
| w | Input | 0.0-1.0 | Inertia weight |
| c1 | Input | 0.0-3.0 | Cognitive coefficient |
| c2 | Input | 0.0-3.0 | Social coefficient |
| Max Iter | Input | 1-500 | Iteration limit |
| Speed | Slider | 1-200ms | Animation speed |

### Buttons
- **▶ Start:** Initialize and begin PSO
- **⏸ Pause:** Pause animation (can resume)
- **🔄 Reset:** Clear all and restart

### Display Areas
- **Canvas:** Visual representation of PSO movement
- **Info Label:** Current iteration and fitness
- **History List:** All saved iterations

### Color Scheme
- **Dark theme:** #1e1e2e (background)
- **Text:** #cdd6f4 (light gray)
- **Goal:** #f38ba8 (pink)
- **Best found:** #a6e3a1 (green)
- **Particles:** 20 distinct colors

---

## 📚 Code Study Guide - Start Here

### Phase 1: Understand Entry Point (5 min)
**File:** `main.py`
```python
import tkinter as tk
from ui import PSOApp

root = tk.Tk()
app = PSOApp(root)
root.mainloop()
```
- Simple bootstrap code
- Creates Tkinter window
- Initializes UI application
- Runs event loop

### Phase 2: Understand UI Structure (10 min)
**File:** `ui.py` - `PSOApp.__init__()` and `_build_ui()`
- How GUI components are created
- Canvas for visualization
- Entry fields for parameters
- Buttons for control
- Listbox for history

### Phase 3: Understand PSO Logic (15 min)
**File:** `pso_logic.py` - `Particle` class
1. `__init__()` - Particle initialization with random velocity
2. `fitness()` - Distance calculation to goal
3. `update_personal_best()` - Check if current position better
4. `update_velocity()` - Apply PSO formula
5. `update_position()` - Move particle

### Phase 4: Understand Algorithm Loop (10 min)
**File:** `pso_logic.py` - `run_pso_step()`
- Evaluates all particles
- Updates personal bests
- Finds global best
- Updates all velocities
- Moves all particles
- Returns new global best

### Phase 5: Understand Animation (10 min)
**File:** `ui.py` - `_animate()` method
- Calls `run_pso_step()` once per frame
- Records history
- Calls `_draw()` to render
- Schedules next frame
- Shows iteration counter and fitness

### Phase 6: Understand Visualization (5 min)
**File:** `ui.py` - `_draw()` method
- Clears canvas
- Draws grid
- Draws goal (pink)
- Draws particles (colored)
- Draws best position (green cross)

### Study Order Recommended
```
1. Run program → observe behavior
2. Read main.py → understand entry
3. Read ui.py _build_ui() → understand controls
4. Read pso_logic.py Particle → understand data
5. Read pso_logic.py run_pso_step() → understand algorithm
6. Read ui.py _animate() → understand loop
7. Read ui.py _draw() → understand rendering
8. Modify parameters → test PSO behavior
```

---

## 🧪 Testing the Project

### Test 1: Verify PSO Convergence
1. Set Goal X = 310, Goal Y = 400
2. Set Particles = 10
3. Click Start
4. Observe fitness decreases over iterations
5. Expected: Fitness starts ~300-400, decreases to 0-10

### Test 2: Verify Speed Control
1. Start PSO
2. Move speed slider left (slow down)
3. Observe animation slows
4. Move slider right (speed up)
5. Observe animation speeds up

### Test 3: Verify History Replay
1. Run PSO for 50 iterations
2. Click iteration 10 in history list
3. Click "📌 Show Selected"
4. Observe particles display at iteration 10 state
5. Repeat for different iterations

### Test 4: Verify Parameter Sensitivity
1. Run with w=0.7 (default) → should converge
2. Run with w=0.1 (low inertia) → may oscillate
3. Run with w=0.95 (high inertia) → more exploration
4. Observe different convergence behaviors

---

## ✨ Key Implementation Highlights

### Why Particles Start on Opposite Side
```python
# Ensures high initial fitness for clear convergence visualization
cx = 50 if gx > 310 else 550    # Opposite side horizontally  
cy = 50 if gy > 325 else 600    # Opposite side vertically
```

### Why Fitness Values Matter
```python
# Low fitness = particle near goal = good solution
# High fitness = particle far from goal = bad solution
best_fitness = 5.32 means particle within 5.32 units of goal
```

### Why History Matters
```python
# Allows replay of any iteration
# Shows PSO convergence over time
# Educational tool to observe algorithm behavior
```

---

## 🎓 Learning Outcomes

After studying this project, you will understand:

✅ **PSO Concepts:**
- Population-based optimization
- Personal best vs global best
- Velocity-based movement
- Convergence behavior

✅ **Algorithm Implementation:**
- How to manually implement PSO
- No library shortcuts - everything from scratch
- Numerical fitness evaluation

✅ **GUI Programming:**
- Tkinter widgets and canvas
- Real-time visualization
- Event-driven programming
- Animation loops

✅ **Software Architecture:**
- Separation of concerns (algorithm vs UI)
- Data flow in interactive applications
- State management
- History/replay patterns

---

## 📝 Summary

This PSO Visualization project demonstrates a complete implementation of Particle Swarm Optimization with:
- ✅ Manual PSO algorithm (no external libraries)
- ✅ Interactive GUI with real-time visualization
- ✅ User-configurable parameters
- ✅ Animation speed control
- ✅ Complete history recording and replay
- ✅ Professional dark-themed interface
- ✅ Comprehensive documentation

**Total Code:** ~375 lines of clean, well-structured Python
**Dependencies:** None (uses built-in Tkinter)
**Learning Value:** High - excellent for understanding optimization algorithms

---

**Ready to study? Start with Phase 1 above and follow the guide! 🚀**
