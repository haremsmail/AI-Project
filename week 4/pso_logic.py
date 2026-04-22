"""
pso_logic.py - PSO Algorithm Implementation (No external library used)

This file contains the Particle class and the PSO update function.
The PSO formula is implemented manually without any AI/optimization library.
"""

import random
import math


class Particle:
    """Represents one particle in the swarm."""

    def __init__(self, x, y, color):
        self.x = x                          # current x position
        self.y = y                          # current y position
        self.vx = random.uniform(-2, 2)     # velocity in x
        self.vy = random.uniform(-2, 2)     # velocity in y
        self.best_x = x                    # personal best x
        self.best_y = y                    # personal best y
        self.best_fitness = float('inf')   # personal best fitness
        self.color = color                 # color for display

    def fitness(self, goal_x, goal_y):
        """Fitness = Euclidean distance to goal (lower is better)."""
        return math.sqrt((self.x - goal_x) ** 2 + (self.y - goal_y) ** 2)

    def update_personal_best(self, goal_x, goal_y):
        """Check if current position is better than personal best."""
        f = self.fitness(goal_x, goal_y)
        if f < self.best_fitness:
            self.best_fitness = f
            self.best_x = self.x
            self.best_y = self.y
        return f

    def update_velocity(self, global_best_x, global_best_y, w, c1, c2):
        """
        PSO velocity update formula:
        v_new = w * v_old + c1 * r1 * (pBest - x) + c2 * r2 * (gBest - x)

        w  = inertia weight
        c1 = cognitive coefficient (pull toward personal best)
        c2 = social coefficient   (pull toward global best)
        r1, r2 = random numbers in [0, 1]
        """
        r1 = random.random()
        r2 = random.random()
        self.vx = (w * self.vx
                   + c1 * r1 * (self.best_x - self.x)
                   + c2 * r2 * (global_best_x - self.x))
        self.vy = (w * self.vy
                   + c1 * r1 * (self.best_y - self.y)
                   + c2 * r2 * (global_best_y - self.y))

    def update_position(self):
        """Move the particle by adding velocity to position."""
        self.x += self.vx
        self.y += self.vy


def run_pso_step(particles, goal_x, goal_y, global_best, w, c1, c2):
    """
    Run ONE iteration of PSO on all particles.

    Parameters:
        particles   - list of Particle objects
        goal_x, goal_y - target position
        global_best - tuple (best_x, best_y, best_fitness)
        w, c1, c2   - PSO parameters

    Returns:
        updated global_best tuple (best_x, best_y, best_fitness)
    """
    gbx, gby, gbf = global_best

    # Step 1: Evaluate fitness and update personal bests
    for p in particles:
        f = p.update_personal_best(goal_x, goal_y)
        if f < gbf:
            gbx, gby, gbf = p.x, p.y, f

    # Step 2: Update velocities and move particles
    for p in particles:
        p.update_velocity(gbx, gby, w, c1, c2)
        p.update_position()

    return (gbx, gby, gbf)
