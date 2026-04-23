#!/usr/bin/env python3
"""Test script to verify PSO fitness calculations match UI behavior"""

from pso_logic import Particle, run_pso_step
import random
import math

COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4", 
          "#f032e6", "#bfef45", "#fabed4", "#469990", "#dcbeff", "#9A6324", 
          "#800000", "#aaffc3", "#808000", "#000075", "#a9a9a9", "#e6beff", "#ffe119", "#ffd8b1"]

# EXACT replication of UI start() logic
params = {
    "n": 10,
    "gx": 310.0,
    "gy": 400.0,
    "w": 0.7,
    "c1": 1.5,
    "c2": 1.5,
    "max_iter": 100
}

particles = []
n = min(params["n"], 20)
cx, cy = random.uniform(50, 570), random.uniform(50, 570)

print(f"Cluster center: ({cx:.1f}, {cy:.1f})")
print(f"Goal: ({params['gx']}, {params['gy']})")
print()

for i in range(n):
    x, y = cx + random.uniform(-40, 40), cy + random.uniform(-40, 40)
    particles.append(Particle(x, y, COLORS[i % len(COLORS)]))
    dist_to_goal = math.sqrt((x - params['gx'])**2 + (y - params['gy'])**2)
    print(f"P{i}: spawn at ({x:7.1f}, {y:7.1f}), dist to goal = {dist_to_goal:7.2f}")

global_best = (params["gx"], params["gy"], float('inf'))

print("\n" + "="*70)
print("PSO Iterations:")
print("="*70)

# EXACT replication of UI _animate() PSO step logic
for iteration in range(1, 11):  # First 10 iterations
    global_best = run_pso_step(particles, params["gx"], params["gy"],
                              global_best, params["w"], params["c1"], params["c2"])
    
    print(f"Iter {iteration:3d} | Global Best Fitness: {global_best[2]:8.2f} | Best at: ({global_best[0]:7.1f}, {global_best[1]:7.1f})")

print("\n✓ If fitness values are > 0 and decreasing, PSO is working!")
print(f"✗ If fitness is 0.00, there's a bug.")
