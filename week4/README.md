# 🎯 PSO Visualization — Particle Swarm Optimization Simulation

A simple, interactive Python application that visualizes the **Particle Swarm Optimization (PSO)** algorithm in real-time. Watch as particles swarm together and converge toward a goal position — all with zero external AI libraries.

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![Tkinter](https://img.shields.io/badge/GUI-Tkinter-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
![No Dependencies](https://img.shields.io/badge/Dependencies-None-brightgreen)

---

## 📖 What is PSO?

**Particle Swarm Optimization** is a nature-inspired optimization algorithm. A group of particles (candidate solutions) fly through a search space, adjusting their positions based on:

- **Personal Best** — the best position each particle has found so far.
- **Global Best** — the best position found by the entire swarm.

Over time, all particles converge toward the optimal solution (the goal).

### PSO Velocity Update Formula

```
v_new = w * v_old + c1 * r1 * (pBest - position) + c2 * r2 * (gBest - position)
```

| Symbol | Meaning |
|--------|---------|
| `w` | Inertia weight — controls how much of the old velocity is kept |
| `c1` | Cognitive coefficient — pull toward personal best |
| `c2` | Social coefficient — pull toward global best |
| `r1, r2` | Random values in [0, 1] for exploration |

---

## ✨ Features

- 🧠 **PSO algorithm implemented from scratch** — no optimization libraries used
- 🎨 **Each particle has a unique color** and number label for easy identification
- 🎯 **Goal position** displayed with a distinct pink marker
- ⚙️ **User-configurable parameters** — number of particles, goal position, w, c1, c2, max iterations
- 🎚️ **Adjustable animation speed** — slow down or speed up to study particle movement
- 📋 **Generation history** — every iteration's best solution and fitness is saved in a list
- 📌 **Replay any iteration** — select a past generation from the list to display it on the canvas
- 🟢 **Global best crosshair** — green marker shows the swarm's best-known position

---

## 📁 Project Structure

```
week4/
├── main.py          # 🚀 Entry point — run this file
├── pso_logic.py     # 🧠 PSO algorithm (Particle class + update formula)
├── ui.py            # 🎨 Tkinter GUI (canvas, controls, animation)
└── README.md        # 📖 This file
```

| File | Responsibility |
|------|---------------|
| `main.py` | Creates the window and starts the app |
| `pso_logic.py` | Contains `Particle` class and `run_pso_step()` — the core algorithm |
| `ui.py` | Builds the GUI, handles user input, runs animation loop, draws particles |

---

## 🚀 How to Run

**Requirements:** Python 3.x (no installation needed — uses only built-in libraries)

```bash
python main.py
```

That's it! No `pip install` required.

---

## 🕹️ How to Use

1. **Set parameters** on the right panel (or keep defaults)
2. Click **▶ Start** to begin the simulation
3. Watch particles move toward the goal
4. Use **⏸ Pause** to stop the animation at any time
5. Adjust the **speed slider** to slow down or speed up the animation
6. Click any entry in the **Generation History** list, then press **📌 Show Selected on Canvas** to replay that iteration
7. Click **🔄 Reset** to clear everything and start over

---

## 🛠️ Technologies Used

- **Python 3** — main programming language
- **Tkinter** — built-in GUI library (no external install)
- **math** — for Euclidean distance calculation
- **random** — for particle initialization and PSO randomness

> ⚠️ **No external library is used for PSO.** The algorithm is implemented manually from the mathematical formula.

---

## 👥 Team

University AI course project — Week 4 Assignment.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
