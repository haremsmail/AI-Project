# 🎯 PSO Visualization — Particle Swarm Optimization

A professional, interactive Python application that visualizes the **Particle Swarm Optimization (PSO)** algorithm in real-time. Watch as a swarm of colorful particles converges toward a goal position using nature-inspired optimization — all implemented **completely from scratch** with **no external AI libraries**.

[![Python](https://img.shields.io/badge/Python-3.7+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![GUI](https://img.shields.io/badge/GUI-Tkinter-green)](https://docs.python.org/3/library/tkinter.html)
[![License](https://img.shields.io/badge/License-MIT-yellow)](#-license)
[![Zero Dependencies](https://img.shields.io/badge/Dependencies-None-brightgreen)](#-setup--installation)

---

## 📖 What is PSO?

**Particle Swarm Optimization** is a nature-inspired algorithm for finding optimal solutions. A swarm of **particles** explores a search space, learning from:

- **Their own success** (personal best position found)
- **The group's success** (global best position found)
- **Their momentum** (velocity/direction)

Over time, particles converge toward the optimal solution.

### Core PSO Equations

**Velocity Update (balance between exploration and exploitation):**
```
v_new = w * v_old + c1 * r1 * (pBest - x) + c2 * r2 * (gBest - x)
```

**Position Update:**
```
x_new = x_old + v_new
```

**Parameter Guide:**
| Parameter | Meaning | Default | Effect |
|-----------|---------|---------|--------|
| **w** | Inertia weight | 0.7 | 0.9 = explore, 0.3 = exploit |
| **c1** | Cognitive coefficient | 1.5 | Pull toward particle's best |
| **c2** | Social coefficient | 1.5 | Pull toward swarm's best |
| **r1, r2** | Random [0,1] | — | Adds randomness for diversity |

---

## ✨ Features

### 🧠 Core Algorithm
- ✅ **100% Manual Implementation** — No external optimization libraries
- ✅ **Proper PSO Formula** — Exactly as defined in research papers
- ✅ **Well-Documented Code** — Every function has docstrings
- ✅ **Object-Oriented** — Clean Particle and PSO classes

### 🎨 Visualization
- ✅ **Real-time Animation** — Smooth particle movement
- ✅ **Velocity Vectors** — Arrows showing direction and speed
- ✅ **Particle Trails** — Faded paths showing movement history
- ✅ **Unique Colors** — 20 distinct particle colors
- ✅ **Goal Marker** — Pink star at target position
- ✅ **Best Position Marker** — Green crosshair at gBest

### 🎮 User Controls
- ✅ **Start/Pause/Reset** — Full control over simulation
- ✅ **Speed Slider** — 1-200 ms delay for perfect pacing
- ✅ **Parameter Configuration** — Adjust algorithm behavior
- ✅ **Generation History** — Full replay capability
- ✅ **Live Statistics** — Real-time iteration and fitness display

### 🎨 Professional UI
- ✅ **Modern Dark Theme** — Easy on the eyes
- ✅ **Responsive Design** — Clean layout
- ✅ **Status Panel** — Shows current state
- ✅ **Grid Reference** — Helps position tracking

---

## 📁 Files & Structure

```
week4/
├── main.py           # 🚀 Entry point (run this!)
├── pso_logic.py      # 🧠 Algorithm implementation
├── ui.py             # 🎨 GUI and visualization
├── requirements.txt  # 📦 Dependencies (none needed)
├── README.md         # 📖 This file
├── SETUP.md          # 🔧 Installation guide
├── QUICKSTART.txt    # ⚡ Quick reference
└── readme.txt        # 🇰🇺 Kurdish guide
```

| File | Lines | Purpose |
|------|-------|---------|
| `main.py` | ~30 | Application entry point |
| `pso_logic.py` | ~160 | Particle class + PSO algorithm |
| `ui.py` | ~500 | Tkinter GUI + animation |
| `requirements.txt` | ~3 | Documentation (no packages) |

---

## 🚀 Quick Start (30 seconds)

### Prerequisites
- Python 3.7+
- Tkinter (built-in)

### Installation
```bash
cd path/to/week4
python main.py
```

**That's it!** No `pip install` needed.

### First Run
1. Click **▶ Start**
2. Watch particles converge toward the goal (pink star)
3. See iteration history update in real-time
4. Click **📌 Show Selected on Canvas** to replay past iterations
5. Adjust speed slider to slow down/speed up

---

## 📖 Complete Usage Guide

### Step 1: Configure Parameters

On the right panel, set:
- **Number of Particles** (1-20): Swarm size
- **Goal X** (0-650): Target X coordinate
- **Goal Y** (0-650): Target Y coordinate
- **Inertia (w)** (0.0-1.0): Momentum control
  - Higher (0.9) = more exploration
  - Lower (0.3) = faster convergence
- **Cognitive (c1)** (0.5-3.0): Personal best weight
- **Social (c2)** (0.5-3.0): Global best weight
- **Max Iterations** (1-1000): Stop condition

### Step 2: Adjust Speed

Use the **Animation Speed** slider (1-200 ms):
- 1 ms = Very fast (hard to follow)
- 50 ms = Default (good balance)
- 200 ms = Slow motion (study behavior)

### Step 3: Run Simulation

Click **▶ Start** button:
- Particles spawn at random cluster
- Animation begins automatically
- History list populates each iteration
- Info label updates in real-time

### Step 4: Control & Monitor

- **⏸ Pause** — Freeze animation
- **🔄 Reset** — Clear everything
- **Speed slider** — Adjust pacing
- **History list** — See all iterations

### Step 5: Replay Iterations

1. Select any row in **Generation History**
2. Click **📌 Show Selected on Canvas**
3. That iteration's state displays on canvas
4. Shows particles, best position, and fitness

### Step 6: Analyze Results

Info display shows:
- Current iteration number
- Best fitness value found
- Number of particles
- Canvas displays:
  - Goal position (🔴 pink)
  - Best position (🟢 green crosshair)
  - All particles with numbers
  - Velocity vectors (arrows)
  - Movement trails (faded lines)

---

## 🧮 Algorithm Deep Dive

### Initialization
```python
# Create N particles
for i in range(num_particles):
    x = random_cluster_center + random_offset
    y = random_cluster_center + random_offset
    v = random(-2, 2) in each dimension
    particle = Particle(x, y, color[i])
```

### Each Iteration
```python
# 1. Evaluate all particles
for particle in swarm:
    fitness = distance(particle.position, goal)
    if fitness < particle.personal_best_fitness:
        particle.personal_best = particle.position
        particle.personal_best_fitness = fitness

# 2. Update global best
global_best = min(all_personal_bests)

# 3. Update velocities (PSO formula)
for particle in swarm:
    r1 = random(0, 1)
    r2 = random(0, 1)
    particle.velocity = (
        w * particle.velocity +
        c1 * r1 * (particle.personal_best - particle.position) +
        c2 * r2 * (global_best - particle.position)
    )

# 4. Update positions
for particle in swarm:
    particle.position += particle.velocity
```

### Convergence
The algorithm terminates when:
- Max iterations reached, OR
- Fitness doesn't improve over many iterations, OR
- Particles cluster too tightly

---

## 🎓 Example Configurations

### Configuration 1: Fast Convergence
Quickly finds a solution (may not be best):
```
w = 0.4      (low momentum)
c1 = 1.0     (forget personal history)
c2 = 2.0     (follow swarm)
Max = 50
```

### Configuration 2: Balanced (DEFAULT)
Good mix of exploration and exploitation:
```
w = 0.7      (moderate momentum)
c1 = 1.5     (remember personal wins)
c2 = 1.5     (follow swarm)
Max = 100
```

### Configuration 3: Thorough Exploration
Explores more, finds better solutions:
```
w = 0.9      (high momentum)
c1 = 2.0     (strong personal learning)
c2 = 1.0     (less social pressure)
Max = 200
```

### Configuration 4: Study Single Particle
Watch pure hill-climbing:
```
Number of Particles = 1
w = 0.7
c1 = 1.5
c2 = 0.0    (no social influence)
Max = 100
```

---

## 💻 Code Architecture

### `Particle` Class (pso_logic.py)

Represents one swarm member:

```python
class Particle:
    def __init__(self, x, y, color)
    def fitness(goal_x, goal_y) → float
    def update_personal_best(goal_x, goal_y)
    def update_velocity(gBest_x, gBest_y, w, c1, c2)
    def update_position()
    def get_speed() → float
```

### `run_pso_step()` (pso_logic.py)

Execute one PSO iteration:

```python
def run_pso_step(particles, goal, best, w, c1, c2):
    # Evaluate and update personal bests
    # Update global best
    # Update velocities
    # Update positions
    return new_global_best
```

### `PSOApp` Class (ui.py)

Main GUI application:

```python
class PSOApp:
    def __init__(root)                 # Setup
    def _build_ui()                    # Create widgets
    def _get_params() → dict           # Read inputs
    def start() / pause() / reset()    # Controls
    def _animate(params)               # Main loop
    def _draw(goal_x, goal_y)          # Render
    def show_selected()                # Replay
```

### Design Principles
- ✅ **Separation of Concerns** — Logic and UI separate
- ✅ **Minimal Dependencies** — Only built-in libraries
- ✅ **Clear Documentation** — Every function explained
- ✅ **Efficient Code** — No unnecessary loops or calculations
- ✅ **Professional Standards** — Follows Python best practices

---

## 🎨 Visual Guide

### Canvas Layout
```
┌──────────────────────────────────────────────┐
│                                              │
│  🔴 GOAL (pink star)                        │
│   ★ ← Target position                       │
│                                             │
│      ╭─────────────╮                        │
│      │ Particles   │   🟢 Best (green +)    │
│      │  (numbered) │                        │
│      │  colors:    │                        │
│      │  🔵🔴🟣 ...  │   
│      │             │ ← Faded                │
│      │ → Velocity  │    movement            │
│      │   vectors   │    trail               │
│      ╰─────────────╯                        │
│                                              │
│  🟢 Best Position  🔴 Goal  → Velocity     │
└──────────────────────────────────────────────┘
650x650 pixels • Dark background • Grid pattern
```

### Control Panel (Right Side)
```
┌─────────────────────────────┐
│ PSO Parameters              │
├─────────────────────────────┤
│ Number of Particles: [10]   │
│ Goal X (0-650):    [325]    │
│ Goal Y (0-650):    [325]    │
│ Inertia (w):       [0.7]    │
│ Cognitive (c1):    [1.5]    │
│ Social (c2):       [1.5]    │
│ Max Iterations:    [100]    │
├─────────────────────────────┤
│ ▶ Start  ⏸ Pause  🔄 Reset │
├─────────────────────────────┤
│ Iteration: 24               │
│ Best Fitness: 12.45         │
├─────────────────────────────┤
│ Generation History          │
│ [Iter  1 | Fit: 412.34]    │
│ [Iter  2 | Fit: 387.12]    │
│ [Iter  3 | Fit: 345.67] ← │
│ ...scrollable...            │
├─────────────────────────────┤
│ 📌 Show Selected on Canvas  │
└─────────────────────────────┘
```

---

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'tkinter'"
**Solution:** Install Tkinter
- Ubuntu: `sudo apt-get install python3-tk`
- Fedora: `sudo dnf install python3-tkinter`
- See SETUP.md for full instructions

### "Python version 2.x is not supported"
**Solution:** Use Python 3.7+
```bash
python3 --version
python3 main.py
```

### Window is too small or cut off
**Solution:** Check display scaling or maximize window
- Window size is 1100x750 pixels
- Designed for 1920x1080+ displays

### Particles not moving
**Solution:** Check these:
1. Did you click ▶ **Start** button?
2. Is Max Iterations >= 1?
3. Is Animation Speed slider > 0?
4. Do all parameters have valid numbers?

### History list shows old data after reset
**Solution:** Click 🔄 **Reset** again, or select and delete rows
- History persists in memory
- Reset clears everything including history

### Simulation finishes too quickly
**Solution:** Increase Max Iterations or adjust parameters
- Default is 100 iterations
- Try 500 or 1000 for longer runs
- Use Pause button to freeze at interesting moments

---

## 📊 Sample Output

Typical convergence process:

```
Iteration  │ Best Fitness  │ Position        │ Convergence
  1        │   412.34      │ (287.12, 298.45)│ ====░░░░░░ 
  5        │   156.78      │ (310.23, 319.87)│ ==========░
 10        │    82.45      │ (318.56, 321.34)│ ==========░
 20        │    34.12      │ (323.45, 324.12)│ ==========░
 50        │     5.89      │ (324.87, 325.02)│ ==========░
100        │     0.03      │ (325.01, 325.02)│ ==========✓
---        │   0.03 ✓      │ Goal reached!   │ Complete
```

---

## 📝 Mathematical Background

### Fitness Metric
We use **Euclidean distance** (to minimize):
```
f(x, y) = √[(x - goal_x)² + (y - goal_y)²]
```

Lower fitness = closer to goal = better solution

### PSO Parameters Effects
| Parameter | High | Low |
|-----------|------|-----|
| **w** | More exploration, overshoots | Fast convergence, may miss optima |
| **c1** | Particles diverge | Particles ignore personal learning |
| **c2** | Herd behavior, local optima | Particles ignore swarm wisdom |

### Convergence Factors
1. **Initial distribution** — Starting cluster affects convergence
2. **Parameter balance** — w, c1, c2 must be tuned
3. **Goal difficulty** — Some goals require more iterations
4. **Random seed** — Each run differs slightly

---

## 🎓 Educational Resources

### What You'll Learn
- ✅ How PSO works (algorithm and code)
- ✅ GUI programming with Tkinter
- ✅ Object-oriented design
- ✅ Real-time visualization techniques
- ✅ Optimization algorithm concepts

### Best Practices Demonstrated
- ✅ Separation of concerns
- ✅ Clean code with docstrings
- ✅ Professional UI design
- ✅ Efficient algorithms
- ✅ User-friendly interface

### Further Study
- Read pso_logic.py for algorithm details
- Read ui.py for visualization techniques
- Experiment with different parameters
- Modify code to add new features

---

## 📄 License

MIT License — Free to use, modify, and distribute

```
Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.
```

---

## 👤 Author

**haremsmail** — AI-Project (Week 4)  
**Version:** 2.0  
**Date:** April 2026

---

## 🔗 Related Files

| File | Read For |
|------|----------|
| SETUP.md | Detailed installation guide |
| QUICKSTART.txt | Quick reference (1 page) |
| readme.txt | Kurdish language guide |
| requirements.txt | Package requirements (none) |

---

## 📞 Support & Questions

### Common Questions

**Q: Do I need to install anything with pip?**  
A: No! Tkinter is built-in. Just run `python main.py`

**Q: Can I modify the code?**  
A: Absolutely! It's MIT licensed. Modify freely for learning.

**Q: How do I export results?**  
A: Check the history display, or modify ui.py to save data.

**Q: Can I add more algorithms?**  
A: Yes! Modify pso_logic.py to add new algorithms or parameter controls.

### For Deeper Understanding

- **Code Comments** — Every function has detailed docstrings
- **README.md** — This file, explains everything
- **SETUP.md** — Installation troubleshooting
- **QUICKSTART.txt** — Quick reference guide

---

## ✨ Features at a Glance

```
✅ Algorithm Implementation
   • 100% manual PSO (no libraries)
   • Proper mathematical formulas
   • Particle class with physics
   • Configurable parameters

✅ Visualization
   • Real-time animation
   • Particle trails and vectors
   • Color-coded particles
   • Goal and best-position markers
   • Grid background

✅ User Interface
   • Professional dark theme
   • Intuitive controls
   • Parameter configuration
   • History replay
   • Live statistics

✅ Code Quality
   • Well-documented
   • Clean architecture
   • Object-oriented design
   • No external dependencies
   • Professional standards
```

---

**🚀 Ready to start? Just run:**
```bash
python main.py
```

**Enjoy watching your swarm converge! 🎯**
