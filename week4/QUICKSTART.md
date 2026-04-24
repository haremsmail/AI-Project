# 🚀 Quick Start Guide - PSO Visualization

## Run the Project

### Windows PowerShell
```powershell
cd c:\Desktop\week1\AI-Project\week4
python main.py
```

### Linux/macOS Terminal
```bash
cd ~/Desktop/week1/AI-Project/week4
python main.py
```

## What You'll See

1. **Main Window** with two panels:
   - **Left:** Canvas showing particles moving toward goal
   - **Right:** Control panel with parameters and history

2. **Initial State:**
   - 10 colored particles on top-left
   - Pink star at goal position (310, 400)
   - Green cross shows best position found

3. **Controls:**
   - Enter particle count, goal position, PSO parameters
   - Click "▶ Start" to begin
   - Use speed slider to adjust animation speed
   - Click "📌 Show Selected" to replay any iteration

## How to Study the Code

### Step 1: Understand the Flow (5 minutes)
Open these files in this order:
1. `main.py` - Entry point (30 lines)
2. `ui.py` - lines 1-30 (GUI initialization)
3. `pso_logic.py` - lines 1-40 (Particle class basics)

### Step 2: Understand PSO Algorithm (10 minutes)
```
Read pso_logic.py in this order:
1. Particle.__init__() - How particles are created
2. Particle.fitness() - How to measure distance to goal
3. Particle.update_velocity() - The PSO formula
4. Particle.update_position() - How particles move
5. run_pso_step() - How one iteration works
```

### Step 3: Understand GUI Animation (10 minutes)
```
Read ui.py:
1. PSOApp.__init__() - GUI setup
2. start() - When user clicks Start button
3. _animate() - Animation loop (THIS IS THE HEART)
4. _draw() - Canvas rendering
```

### Step 4: Trace One Complete Iteration (10 minutes)
Run this Python code to see PSO in action:
```python
from pso_logic import Particle, run_pso_step
import random

# Create 3 particles
particles = []
for i in range(3):
    x = 100 + random.uniform(-20, 20)
    y = 100 + random.uniform(-20, 20)
    particles.append(Particle(x, y, "#fff"))

# Run ONE PSO iteration
goal = (310, 400)
gb = (goal[0], goal[1], float('inf'))
gb = run_pso_step(particles, goal[0], goal[1], gb, 0.7, 1.5, 1.5)

# See results
print(f"Best fitness: {gb[2]:.2f}")
print(f"Particles moved to new positions")
```

## Verify All Requirements Are Met

| Requirement | Location | Status |
|-------------|----------|--------|
| Python main logic | `pso_logic.py` | ✅ |
| Random particle placement | `ui.py` line 93-97 | ✅ |
| User parameters | GUI input fields | ✅ |
| Goal in different color | `ui.py` line 146-148 | ✅ |
| Animation speed | Speed slider | ✅ |
| Save history | `ui.py` line 124-127 | ✅ |
| Particle differentiation | 20 colors + numbering | ✅ |
| No PSO libraries | `pso_logic.py` manual formula | ✅ |

## Key Code Sections

### PSO Formula (The Most Important Part)
**File:** `pso_logic.py` lines 87-103

```python
def update_velocity(self, global_best_x, global_best_y, w, c1, c2):
    r1 = random.random()
    r2 = random.random()
    
    # PSO velocity update formula
    self.vx = (w * self.vx
               + c1 * r1 * (self.best_x - self.x)
               + c2 * r2 * (global_best_x - self.x))
    
    self.vy = (w * self.vy
               + c1 * r1 * (self.best_y - self.y)
               + c2 * r2 * (global_best_y - self.y))
```

### Animation Loop (The Core Loop)
**File:** `ui.py` lines 114-133

```python
def _animate(self, params):
    # 1. Run ONE PSO step
    self.global_best = run_pso_step(self.particles, params["gx"], 
                                    params["gy"], self.global_best, 
                                    params["w"], params["c1"], params["c2"])
    
    # 2. Save to history
    self.history.append({...})
    
    # 3. Render canvas
    self._draw(params["gx"], params["gy"])
    
    # 4. Schedule next frame
    self.root.after(self.speed_var.get(), lambda: self._animate(params))
```

## Common Questions

**Q: Why do particles start on the top-left?**
A: They spawn on opposite side of canvas from goal for clear convergence visualization.

**Q: Can I change the goal while running?**
A: No, click "🔄 Reset" first, change parameters, then "▶ Start" again.

**Q: What do the fitness numbers mean?**
A: Fitness = distance from particle to goal. Lower is better. Goal = 0.

**Q: Can particles go outside canvas?**
A: Yes, PSO doesn't enforce boundaries. You can set Goal X to 50 or 600 to test.

**Q: Why does fitness sometimes stay the same?**
A: Particles found the best they can and are not improving further (convergence).

## Next Steps

1. **Understand:** Read `PROJECT_ANALYSIS.md` for complete details
2. **Experiment:** Change PSO parameters (w, c1, c2) and see convergence differences
3. **Modify:** Add stopping condition when fitness < threshold
4. **Extend:** Add inertia weight decay for faster convergence

---

**All requirements from Assignment D are ✅ COMPLETE!**

See `PROJECT_ANALYSIS.md` for detailed verification of each requirement.
