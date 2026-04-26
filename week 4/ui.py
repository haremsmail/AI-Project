"""
ui.py - GUI for PSO Visualization

This file builds the tkinter interface: canvas, controls, and history list.
It imports the PSO logic from pso_logic.py.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import random
from pso_logic import Particle, run_pso_step


# Distinct colors so each particle is easy to identify
COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9A6324", "#800000", "#aaffc3", "#808000",
    "#000075", "#a9a9a9", "#e6beff", "#ffe119", "#ffd8b1",
]


class PSOApp:
    """Main application window for PSO simulation."""

    def __init__(self, root):
        self.root = root
        self.root.title("PSO Visualization - Particle Swarm Optimization")
        self.root.geometry("1050x700")
        self.root.configure(bg="#1e1e2e")
        self.root.resizable(False, False)

        # State variables
        self.particles = []
        self.running = False
        self.iteration = 0
        self.history = []   # stores snapshots of each iteration
        self.global_best = (0, 0, float('inf'))

        self._build_ui()

    # ----------------------------------------------------------------
    # BUILD THE USER INTERFACE
    # ----------------------------------------------------------------

    def _build_ui(self):
        """Create all widgets."""
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TLabel", background="#1e1e2e",
                         foreground="#cdd6f4", font=("Arial", 10))
        style.configure("Header.TLabel", font=("Arial", 13, "bold"),
                         foreground="#89b4fa", background="#1e1e2e")

        # --- Canvas (left side) ---
        self.canvas = tk.Canvas(
            self.root, width=650, height=650, bg="#181825",
            highlightthickness=1, highlightbackground="#45475a")
        self.canvas.place(x=10, y=10)

        # --- Controls panel (right side) ---
        ctrl = tk.Frame(self.root, bg="#1e1e2e")
        ctrl.place(x=680, y=10, width=360, height=680)

        ttk.Label(ctrl, text="PSO Parameters",
                  style="Header.TLabel").pack(pady=(0, 8))

        # Parameter input fields
        fields = [
            ("Number of Particles:", "num_particles", "10"),
            ("Goal X (0-650):",      "goal_x",        "325"),
            ("Goal Y (0-650):",      "goal_y",        "325"),
            ("Inertia (w):",         "w",             "0.7"),
            ("Cognitive (c1):",      "c1",            "1.5"),
            ("Social (c2):",         "c2",            "1.5"),
            ("Max Iterations:",      "max_iter",      "100"),
        ]
        self.entries = {}
        for label_text, key, default in fields:
            row = tk.Frame(ctrl, bg="#1e1e2e")
            row.pack(fill="x", padx=5, pady=2)
            ttk.Label(row, text=label_text, width=20,
                      anchor="w").pack(side="left")
            entry = tk.Entry(row, width=10, bg="#313244", fg="#cdd6f4",
                             insertbackground="#cdd6f4",
                             font=("Arial", 10), relief="flat")
            entry.insert(0, default)
            entry.pack(side="left", padx=5)
            self.entries[key] = entry

        # Animation speed slider
        ttk.Label(ctrl, text="Animation Speed").pack(pady=(10, 0))
        self.speed_var = tk.IntVar(value=50)
        self.speed_slider = tk.Scale(
            ctrl, from_=1, to=200, orient="horizontal",
            variable=self.speed_var, bg="#1e1e2e", fg="#cdd6f4",
            highlightthickness=0, troughcolor="#313244",
            label="Delay (ms)", font=("Arial", 9))
        self.speed_slider.pack(fill="x", padx=10)

        # Start / Pause / Reset buttons
        btn_frame = tk.Frame(ctrl, bg="#1e1e2e")
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="▶ Start", command=self.start,
                  bg="#a6e3a1", fg="#1e1e2e",
                  font=("Arial", 10, "bold"), width=8,
                  relief="flat").pack(side="left", padx=3)
        tk.Button(btn_frame, text="⏸ Pause", command=self.pause,
                  bg="#f9e2af", fg="#1e1e2e",
                  font=("Arial", 10, "bold"), width=8,
                  relief="flat").pack(side="left", padx=3)
        tk.Button(btn_frame, text="🔄 Reset", command=self.reset,
                  bg="#f38ba8", fg="#1e1e2e",
                  font=("Arial", 10, "bold"), width=8,
                  relief="flat").pack(side="left", padx=3)

        # Current iteration info
        self.info_label = tk.Label(
            ctrl, text="Iteration: 0 | Best Fitness: -",
            bg="#1e1e2e", fg="#89b4fa", font=("Arial", 10, "bold"))
        self.info_label.pack(pady=5)

        # Generation history list
        ttk.Label(ctrl, text="Generation History",
                  style="Header.TLabel").pack(pady=(5, 3))
        list_frame = tk.Frame(ctrl, bg="#1e1e2e")
        list_frame.pack(fill="both", expand=True, padx=5, pady=(0, 5))

        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side="right", fill="y")
        self.history_list = tk.Listbox(
            list_frame, bg="#313244", fg="#cdd6f4",
            font=("Consolas", 9), selectbackground="#89b4fa",
            selectforeground="#1e1e2e", relief="flat",
            yscrollcommand=scrollbar.set)
        self.history_list.pack(fill="both", expand=True)
        scrollbar.config(command=self.history_list.yview)

        # Button to display a selected generation on canvas
        tk.Button(ctrl, text="📌 Show Selected on Canvas",
                  command=self.show_selected,
                  bg="#89b4fa", fg="#1e1e2e",
                  font=("Arial", 10, "bold"),
                  relief="flat").pack(pady=5)

    # ----------------------------------------------------------------
    # READ USER INPUTS
    # ----------------------------------------------------------------

    def _get_params(self):
        """Read and validate all parameter fields."""
        try:
            return {
                "n":        int(self.entries["num_particles"].get()),
                "gx":       float(self.entries["goal_x"].get()),
                "gy":       float(self.entries["goal_y"].get()),
                "w":        float(self.entries["w"].get()),
                "c1":       float(self.entries["c1"].get()),
                "c2":       float(self.entries["c2"].get()),
                "max_iter": int(self.entries["max_iter"].get()),
            }
        except ValueError:
            messagebox.showerror("Input Error",
                                 "Please enter valid numbers for all fields.")
            return None

    # ----------------------------------------------------------------
    # CONTROL BUTTONS
    # ----------------------------------------------------------------

    def start(self):
        """Start or resume the simulation."""
        if self.running:
            return
        params = self._get_params()
        if not params:
            return

        # Create particles on first start
        if not self.particles:
            n = min(params["n"], 20)
            # Place all particles near a random cluster center
            cx = random.uniform(50, 600)
            cy = random.uniform(50, 600)
            for i in range(n):
                color = COLORS[i % len(COLORS)]
                x = cx + random.uniform(-40, 40)
                y = cy + random.uniform(-40, 40)
                self.particles.append(Particle(x, y, color))
            self.global_best = (cx, cy, float('inf'))
            self.iteration = 0
            self.history = []
            self.history_list.delete(0, tk.END)

        self.running = True
        self._animate(params)

    def pause(self):
        """Pause the animation."""
        self.running = False

    def reset(self):
        """Clear everything and start fresh."""
        self.running = False
        self.particles = []
        self.iteration = 0
        self.history = []
        self.global_best = (0, 0, float('inf'))
        self.history_list.delete(0, tk.END)
        self.info_label.config(text="Iteration: 0 | Best Fitness: -")
        self.canvas.delete("all")

    # ----------------------------------------------------------------
    # ANIMATION LOOP
    # ----------------------------------------------------------------

    def _animate(self, params):
        """Run one PSO step, draw, then schedule the next frame."""
        if not self.running:
            return
        if self.iteration >= params["max_iter"]:
            self.running = False
            messagebox.showinfo(
                "Done",
                f"Finished! Best fitness: {self.global_best[2]:.4f}")
            return

        # Run one PSO iteration (calls pso_logic)
        self.global_best = run_pso_step(
            self.particles,
            params["gx"], params["gy"],
            self.global_best,
            params["w"], params["c1"], params["c2"]
        )
        self.iteration += 1

        # Save snapshot for history
        snapshot = [(p.x, p.y, p.color) for p in self.particles]
        best_fit = self.global_best[2]
        self.history.append({
            "iter":    self.iteration,
            "best_x":  self.global_best[0],
            "best_y":  self.global_best[1],
            "fitness": best_fit,
            "positions": snapshot
        })
        self.history_list.insert(
            tk.END,
            f"Iter {self.iteration:3d} | Fit: {best_fit:8.2f} "
            f"| Pos: ({self.global_best[0]:.0f}, {self.global_best[1]:.0f})")

        # Draw current state
        self._draw(params["gx"], params["gy"])

        # Update info label
        self.info_label.config(
            text=f"Iteration: {self.iteration} | Best Fitness: {best_fit:.4f}")

        # Schedule next frame based on speed slider
        delay = self.speed_var.get()
        self.root.after(delay, lambda: self._animate(params))

    # ----------------------------------------------------------------
    # DRAWING
    # ----------------------------------------------------------------

    def _draw(self, goal_x, goal_y):
        """Render particles, goal, and global best on the canvas."""
        self.canvas.delete("all")

        # Grid lines
        for i in range(0, 651, 50):
            self.canvas.create_line(i, 0, i, 650, fill="#313244")
            self.canvas.create_line(0, i, 650, i, fill="#313244")

        # Goal marker (pink circle with star)
        r = 14
        self.canvas.create_oval(
            goal_x - r, goal_y - r, goal_x + r, goal_y + r,
            fill="#f38ba8", outline="#fab387", width=3)
        self.canvas.create_text(
            goal_x, goal_y, text="★",
            fill="#1e1e2e", font=("Arial", 14, "bold"))
        self.canvas.create_text(
            goal_x, goal_y - 22, text="GOAL",
            fill="#f38ba8", font=("Arial", 9, "bold"))

        # Particles (colored circles with number labels)
        for i, p in enumerate(self.particles):
            sz = 8
            self.canvas.create_oval(
                p.x - sz, p.y - sz, p.x + sz, p.y + sz,
                fill=p.color, outline="white", width=1)
            self.canvas.create_text(
                p.x, p.y, text=str(i + 1),
                fill="white", font=("Arial", 7, "bold"))

        # Global best crosshair
        gbx, gby = self.global_best[0], self.global_best[1]
        self.canvas.create_line(gbx - 8, gby, gbx + 8, gby,
                                fill="#a6e3a1", width=2)
        self.canvas.create_line(gbx, gby - 8, gbx, gby + 8,
                                fill="#a6e3a1", width=2)

    # ----------------------------------------------------------------
    # SHOW SELECTED GENERATION
    # ----------------------------------------------------------------

    def show_selected(self):
        """Display a past generation's particle positions on the canvas."""
        sel = self.history_list.curselection()
        if not sel:
            messagebox.showinfo("Select",
                                "Please select a generation from the list.")
            return

        idx = sel[0]
        record = self.history[idx]
        params = self._get_params()
        if not params:
            return

        self.canvas.delete("all")

        # Grid
        for i in range(0, 651, 50):
            self.canvas.create_line(i, 0, i, 650, fill="#313244")
            self.canvas.create_line(0, i, 650, i, fill="#313244")

        # Goal
        gx, gy = params["gx"], params["gy"]
        r = 14
        self.canvas.create_oval(
            gx - r, gy - r, gx + r, gy + r,
            fill="#f38ba8", outline="#fab387", width=3)
        self.canvas.create_text(gx, gy, text="★",
                                fill="#1e1e2e", font=("Arial", 14, "bold"))
        self.canvas.create_text(gx, gy - 22, text="GOAL",
                                fill="#f38ba8", font=("Arial", 9, "bold"))

        # Particles from snapshot
        for i, (px, py, color) in enumerate(record["positions"]):
            sz = 8
            self.canvas.create_oval(
                px - sz, py - sz, px + sz, py + sz,
                fill=color, outline="white", width=1)
            self.canvas.create_text(
                px, py, text=str(i + 1),
                fill="white", font=("Arial", 7, "bold"))

        # Best position crosshair
        bx, by = record["best_x"], record["best_y"]
        self.canvas.create_line(bx - 8, by, bx + 8, by,
                                fill="#a6e3a1", width=2)
        self.canvas.create_line(bx, by - 8, bx, by + 8,
                                fill="#a6e3a1", width=2)

        # Title text
        self.canvas.create_text(
            325, 15,
            text=f"Iteration {record['iter']} — Fitness: {record['fitness']:.4f}",
            fill="#89b4fa", font=("Arial", 11, "bold"))
