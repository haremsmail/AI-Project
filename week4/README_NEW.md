# 🎯 PSO Visualization — Particle Swarm Optimization

A professional, interactive Python application that visualizes the **Particle Swarm Optimization (PSO)** algorithm in real-time. Watch as a swarm of colorful particles converges toward a goal position using nature-inspired optimization — all implemented **completely from scratch** with **no external AI libraries**.

[![Python](https://img.shields.io/badge/Python-3.7+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![GUI](https://img.shields.io/badge/GUI-Tkinter-green)](https://docs.python.org/3/library/tkinter.html)
[![License](https://img.shields.io/badge/License-MIT-yellow)](#license)
[![Zero Dependencies](https://img.shields.io/badge/Dependencies-Tkinter%20Only-brightgreen)](#-setup--installation)

---

## 📖 What is PSO (Particle Swarm Optimization)?

**Particle Swarm Optimization** is a nature-inspired metaheuristic algorithm for solving optimization problems. A group of **particles** (candidate solutions) fly through a search space, adjusting their positions based on 3 key influences:

1. **Personal Best (pBest)** — The best position this particle has ever found
2. **Global Best (gBest)** — The best position found by the entire swarm
3. **Inertia** — Tendency to keep moving in the same direction

### The PSO Velocity Update Formula

The heart of PSO—particles update their velocity at each iteration:

```
v_new = w * v_old + c1 * r1 * (pBest - x) + c2 * r2 * (gBest - x)
```

**Explanation:**
- **w** — Inertia weight (0.0–1.0): Controls how much of the old velocity is retained. Higher = more exploration, lower = more exploitation.
- **c1** — Cognitive coefficient (~1.5): Pull toward personal best. Encourages particles to remember good solutions.
- **c2** — Social coefficient (~1.5): Pull toward global best. Encourages particles to follow the swarm.
- **r1, r2** — Random values in [0, 1]: Adds stochasticity for exploration.

### Position Update

```
x_new = x_old + v_new
```

---

## ✨ Features

### Core Features
- 🧠 **100% Manual PSO Implementation** — Algorithm written entirely in Python, no optimization libraries
- 🎨 **Colorful Particle Swarm** — Each particle has a unique color and number label
- 🎯 **Goal Marker** — Pink star shows the target position
- 🟢 **Best Position Indicator** — Green crosshair shows swarm's best-known solution
- ⚙️ **Fully Configurable Parameters** — Adjust particles, goal, w, c1, c2, max iterations
- 🎚️ **Speed Control Slider** — 1–200 ms delay for perfect visualization

### Visualization Features
- 📊 **Velocity Vectors** — Arrows show each particle's direction and speed
- 🌌 **Particle Trails** — Faded paths showing recent movement history
- 📋 **Generation History List** — Every iteration saved with fitness and position
- 📌 **Replay Functionality** — Click any iteration to display that moment
- 🎬 **Smooth Animation** — Real-time rendering
- 🔲 **Grid Background** — Reference grid for position tracking

### Control Features
- ▶️ **Start Button** — Begin or resume
- ⏸️ **Pause Button** — Freeze animation
- 🔄 **Reset Button** — Clear and restart
- 💾 **Live Statistics** — Real-time iteration and fitness display

---

## 📁 Project Structure

```
week4/
├── main.py                 # Entry point — run this to start
├── ui.py                   # GUI implementation (Tkinter)
├── pso_logic.py           # PSO algorithm + Particle class
├── requirements.txt       # Dependencies
├── README.md              # This file
└── readme.txt             # Kurdish description
```

---

## 🚀 Setup & Installation

### Prerequisites
- **Python 3.7 or higher**
- **Tkinter** (included with Python by default)

### Step 1: Verify Python
```bash
python --version
```

### Step 2: Verify Tkinter
```bash
python -c "import tkinter; print('Tkinter is installed!')"
```

### Step 3: If Tkinter Missing
- **Ubuntu/Debian:** `sudo apt-get install python3-tk`
- **Fedora:** `sudo dnf install python3-tkinter`
- **macOS:** Already included with python.org Python
- **Windows:** Already included by default

### Step 4: Run
```bash
cd path/to/week4
python main.py
```

---

## 📖 How to Use

### 1. Configure PSO Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Number of Particles | 10 | 1–20 | Swarm size |
| Goal X | 325 | 0–650 | Target X |
| Goal Y | 325 | 0–650 | Target Y |
| Inertia (w) | 0.7 | 0.0–1.0 | Momentum weight |
| Cognitive (c1) | 1.5 | 0.5–3.0 | Personal best pull |
| Social (c2) | 1.5 | 0.5–3.0 | Global best pull |
| Max Iterations | 100 | 1–1000 | Stop condition |

### 2. Adjust Animation Speed
Slider controls frame delay (1–200 ms)

### 3. Start
Click **▶ Start** — particles spawn near random cluster and converge

### 4. Control
- **⏸ Pause** — Freeze
- **🔄 Reset** — Clear and restart

### 5. Inspect History
Select any iteration and click **📌 Show Selected on Canvas** to replay

### 6. Interpret Visualization
- 🟢 **Green Crosshair** = Best found position
- 🔴 **Pink Circle** = Goal
- **Colored Numbers** = Particles
- **Arrows** = Velocity vectors
- **Faded Rays** = Movement trails

---

## 🧮 Algorithm Explanation

### Initialization
1. N particles spawn randomly near a cluster center
2. Each gets random initial velocity
3. Personal and global bests initialized

### Each Iteration
```
1. Evaluate fitness for each particle
2. Update particle personal bests
3. Update global best
4. Update velocities using PSO formula
5. Update positions
```

### Convergence
Particles gradually converge toward global best through balance of:
- Inertia (maintain direction)
- Cognitive (remember personal success)
- Social (follow swarm success)

---

## 🎓 Example Configurations

### Fast Convergence
```
w=0.4, c1=1.0, c2=2.0, iterations=50
```

### Balanced
```
w=0.7, c1=1.5, c2=1.5, iterations=100
```

### Exploration
```
w=0.9, c1=2.0, c2=1.0, iterations=200
```

---

## 💻 Code Architecture

### `Particle` Class (pso_logic.py)
- `__init__()` — Initialize position, velocity, color
- `fitness()` — Calculate distance to goal
- `update_personal_best()` — Update best known position
- `update_velocity()` — Apply PSO formula
- `update_position()` — Move particle
- `get_speed()` — Return velocity magnitude

### `PSOApp` Class (ui.py)
- `__init__()` — Setup GUI
- `_build_ui()` — Create widgets
- `_get_params()` — Read user inputs
- `start()` / `pause()` / `reset()` — Control buttons
- `_animate()` — Main loop
- `_draw()` — Render canvas
- `show_selected()` — Replay iteration

### `run_pso_step()` (pso_logic.py)
Core PSO algorithm execution

---

## 🎨 UI Theme

Modern dark theme:
- **Background:** `#1e1e2e`
- **Info (Blue):** `#89b4fa`
- **Success (Green):** `#a6e3a1`
- **Goal (Pink):** `#f38ba8`
- **Accent (Yellow):** `#f9e2af`

---

## 🐛 Troubleshooting

**Window too small?** → Check display scaling

**Particles not moving?** → Increase max iterations, check animation speed > 0

**History shows old data?** → Click Reset to clear all

**Import error?** → Ensure Tkinter installed (see Setup section)

---

## 📊 Sample Run

```
▶ Start
├─ Iter 1   | Fit: 412.34 | Pos: (287, 298)
├─ Iter 5   | Fit: 156.78 | Pos: (310, 319)
├─ Iter 10  | Fit: 82.45  | Pos: (318, 322)
├─ Iter 50  | Fit: 5.89   | Pos: (325, 325)
└─ Iter 100 | Fit: 0.03   | Pos: (325.01, 325.02)
✓ Complete
```

---

## 📝 Mathematical Background

### Fitness Function
**Euclidean distance** (minimized):
```
f(x,y) = sqrt((x-goal_x)² + (y-goal_y)²)
```

### Parameter Effects
- **w too high** → Overshooting, slow convergence
- **w too low** → Premature convergence
- **c1 >> c2** → Divergence
- **c2 >> c1** → Local optima
- **Balanced** → Optimal convergence

---

## 🎓 Educational Value

Perfect for:
- 📚 University assignments
- 🧑‍💻 Portfolio projects
- 🎓 Learning optimization algorithms
- 🎨 GUI design practice

---

## 📄 License

MIT License — Free to use and modify

---

## 👤 Author

**haremsmail** — AI-Project (Week 4)

---

## 📞 Support

All functions have detailed docstrings. Experiment with parameters to understand PSO behavior!

**Enjoy! 🚀**

*Version: 2.0 (Enhanced with velocity vectors, trails, professional UI)*
