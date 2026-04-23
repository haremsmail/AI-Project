"""
pso_logic.py - PSO Algorithm Implementation (No external library used)

This file contains the Particle class and the PSO update function.
The PSO formula is implemented manually without any AI/optimization library.
"""

import random
import math


class Particle:
    """
    Represents one particle in the PSO swarm.
    
    Each particle has:
    - Current position (x, y) in 2D space
    - Current velocity (vx, vy) 
    - Personal best position found so far
    - A unique color for visualization
    """

    def __init__(self, x, y, color):
        """
        Initialize a particle at position (x, y) with given color.
        Velocity is initialized randomly to provide initial exploration.
        """
        self.x = x                          # current x position
        self.y = y                          # current y position
        self.vx = random.uniform(-2, 2)     # velocity in x direction
        self.vy = random.uniform(-2, 2)     # velocity in y direction
        self.best_x = x                     # personal best x position
        self.best_y = y                     # personal best y position
        self.best_fitness = float('inf')    # personal best fitness score
        self.color = color                  # unique color for display

    def fitness(self, goal_x, goal_y):
        """
        Calculate fitness as Euclidean distance to goal.
        
        In PSO, we minimize fitness (lower is better).
        Fitness = sqrt((x - goal_x)^2 + (y - goal_y)^2)
        
        Args:
            goal_x, goal_y: Target coordinates in 2D space
            
        Returns:
            Fitness value (distance to goal, lower is better)
        """
        return math.sqrt((self.x - goal_x) ** 2 + (self.y - goal_y) ** 2)

    def update_personal_best(self, goal_x, goal_y):
        """
        Check if current position is better than the particle's personal best.
        If so, update the personal best position and fitness.
        
        Args:
            goal_x, goal_y: Target coordinates
            
        Returns:
            Current fitness value
        """
        f = self.fitness(goal_x, goal_y)
        if f < self.best_fitness:
            self.best_fitness = f
            self.best_x = self.x
            self.best_y = self.y
        return f

    def update_velocity(self, global_best_x, global_best_y, w, c1, c2):
        """
        Update particle velocity using the PSO velocity formula:
        
        v_new = w*v_old + c1*r1*(pBest - x) + c2*r2*(gBest - x)
        
        This formula balances three forces:
        1. Inertia (w*v_old): Keep moving in the same direction
        2. Cognitive (c1*r1*(pBest-x)): Pull toward personal best
        3. Social (c2*r2*(gBest-x)): Pull toward global best
        
        Args:
            global_best_x, global_best_y: Best position found by swarm
            w: Inertia weight (0.0 - 1.0) — higher values = more momentum
            c1: Cognitive coefficient (typically 1.5) — pull toward personal best
            c2: Social coefficient (typically 1.5) — pull toward global best
        """
        # Random values for stochastic exploration
        r1 = random.random()
        r2 = random.random()
        
        # Update x-velocity
        self.vx = (w * self.vx
                   + c1 * r1 * (self.best_x - self.x)
                   + c2 * r2 * (global_best_x - self.x))
        
        # Update y-velocity
        self.vy = (w * self.vy
                   + c1 * r1 * (self.best_y - self.y)
                   + c2 * r2 * (global_best_y - self.y))

    def update_position(self):
        """
        Move the particle by adding velocity to current position.
        
        x_new = x + v
        y_new = y + v
        """
        self.x += self.vx
        self.y += self.vy
    
    def get_speed(self):
        """Calculate the magnitude of velocity (particle speed)."""
        return math.sqrt(self.vx ** 2 + self.vy ** 2)


def run_pso_step(particles, goal_x, goal_y, global_best, w, c1, c2):
    """
    Execute ONE complete iteration of the PSO algorithm.
    
    This function performs the core PSO operations:
    1. Evaluate fitness of each particle
    2. Update personal best positions
    3. Track global best across the swarm
    4. Update all particle velocities based on PSO formula
    5. Move all particles to new positions
    
    Args:
        particles (list): List of Particle objects in the swarm
        goal_x, goal_y (float): Target position coordinates
        global_best (tuple): Current best state (best_x, best_y, best_fitness)
        w (float): Inertia weight — controls momentum/exploration tradeoff
        c1 (float): Cognitive coefficient — emphasis on personal best
        c2 (float): Social coefficient — emphasis on global best
    
    Returns:
        tuple: Updated global_best state (best_x, best_y, best_fitness)
        
    PSO Algorithm Flow:
        For each iteration:
        1. Evaluate all particles
        2. Update pBest (personal best) if current position is better
        3. Update gBest (global best) if any pBest is better than gBest
        4. Update velocities using PSO formula with pBest and gBest
        5. Update positions
    """
    gbx, gby, gbf = global_best

    # ---- Step 1 & 2: Evaluate fitness and update personal bests ----
    for p in particles:
        f = p.update_personal_best(goal_x, goal_y)
        # Update global best if this particle found a better solution
        if f < gbf:
            gbx, gby, gbf = p.x, p.y, f

    # ---- Step 3 & 4: Update velocities and positions ----
    for p in particles:
        p.update_velocity(gbx, gby, w, c1, c2)
        p.update_position()

    return (gbx, gby, gbf)
