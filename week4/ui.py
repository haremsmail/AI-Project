"""ui.py - PSO Visualization GUI (Compact Version)"""
import tkinter as tk
from tkinter import ttk, messagebox
import random
""" the code is ui"""

from pso_logic import Particle, run_pso_step

COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4", 
          "#f032e6", "#bfef45", "#fabed4", "#469990", "#dcbeff", "#9A6324", 
          "#800000", "#aaffc3", "#808000", "#000075", "#a9a9a9", "#e6beff", "#ffe119", "#ffd8b1"]

class PSOApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🎯 PSO Visualization")
        self.root.geometry("1000x700")
        self.root.configure(bg="#1e1e2e")
        self.root.resizable(False, False)
        self.particles = []
        self.running = False
        self.iteration = 0
        self.history = []
        self.global_best = (0, 0, float('inf'))
        """ gloable wata x postion y postion"""
        self._build_ui()

    def _build_ui(self):
        self.canvas = tk.Canvas(self.root, width=620, height=650, bg="#181825", 
                               highlightthickness=1, highlightbackground="#45475a")
        """ au du parametery kotay borderthinkess bordercoloer"""
        self.canvas.place(x=10, y=10)
        
        ctrl = tk.Frame(self.root, bg="#1e1e2e")
        ctrl.place(x=650, y=10, width=340, height=680)
        """ x,y mabasty move right lefta"""
        
        ttk.Label(ctrl, text="PSO Parameters").pack(pady=5)
        
        self.entries = {}
        for label, key, default in [("Particles:", "n", "10"), ("Goal X:", "gx", "310"), 
                                     ("Goal Y:", "gy", "325"), ("w:", "w", "0.7"), 
                                     ("c1:", "c1", "1.5"), ("c2:", "c2", "1.5"), ("Max Iter:", "iter", "100")]:
            f = tk.Frame(ctrl, bg="#1e1e2e")
            f.pack(fill="x", padx=5, pady=2)
            ttk.Label(f, text=label, width=12).pack(side="left")
            e = tk.Entry(f, width=12, bg="#313244", fg="#cdd6f4", font=("Arial", 9), relief="flat")
            e.insert(0, default)
            e.pack(side="left", padx=5)
            self.entries[key] = e
        
        ttk.Label(ctrl, text="Speed (ms)").pack(pady=5)
        self.speed_var = tk.IntVar(value=50)
        tk.Scale(ctrl, from_=1, to=200, orient="h", variable=self.speed_var, 
                bg="#1e1e2e", fg="#cdd6f4", highlightthickness=0, troughcolor="#313244").pack(fill="x", padx=10)
        
        btn_f = tk.Frame(ctrl, bg="#1e1e2e")
        btn_f.pack(pady=8)
        tk.Button(btn_f, text="▶ Start", command=self.start, bg="#a6e3a1", fg="#1e1e2e", 
                 font=("Arial", 9, "bold"), width=7, relief="flat").pack(side="left", padx=2)
        tk.Button(btn_f, text="⏸ Pause", command=self.pause, bg="#f9e2af", fg="#1e1e2e", 
                 font=("Arial", 9, "bold"), width=7, relief="flat").pack(side="left", padx=2)
        tk.Button(btn_f, text="🔄 Reset", command=self.reset, bg="#f38ba8", fg="#1e1e2e", 
                 font=("Arial", 9, "bold"), width=7, relief="flat").pack(side="left", padx=2)
        
        self.info = tk.Label(ctrl, text="Iter: 0 | Fit: -", bg="#1e1e2e", fg="#89b4fa", font=("Arial", 9, "bold"))
        self.info.pack(pady=5)
        
        ttk.Label(ctrl, text="History").pack(pady=(5, 2))
        f_list = tk.Frame(ctrl, bg="#1e1e2e")
        f_list.pack(fill="both", expand=True, padx=5, pady=(0, 5))
        scroll = tk.Scrollbar(f_list)
        scroll.pack(side="right", fill="y")
        self.history_list = tk.Listbox(f_list, bg="#313244", fg="#cdd6f4", font=("Consolas", 8), 
                                       yscrollcommand=scroll.set, relief="flat")
        self.history_list.pack(fill="both", expand=True)
        scroll.config(command=self.history_list.yview)
        
        tk.Button(ctrl, text="📌 Show Selected", command=self.show_selected, bg="#89b4fa", 
                 fg="#1e1e2e", font=("Arial", 9, "bold"), relief="flat").pack(pady=5)

    def _get_params(self):
        try:
            return {"n": int(self.entries["n"].get()), "gx": float(self.entries["gx"].get()),
                   "gy": float(self.entries["gy"].get()), "w": float(self.entries["w"].get()),
                   "c1": float(self.entries["c1"].get()), "c2": float(self.entries["c2"].get()),
                   "max_iter": int(self.entries["iter"].get())}
        except ValueError:
            messagebox.showerror("Error", "Invalid input")
            return None

    def start(self):
        if self.running: return
        params = self._get_params()
        if not params: return
        if not self.particles:
            n = min(params["n"], 20)
            # Spawn cluster on OPPOSITE side of canvas from goal for better visualization
            gx, gy = params["gx"], params["gy"]
            cx = 50 if gx > 310 else 550  # Opposite side horizontally
            cy = 50 if gy > 325 else 600  # Opposite side vertically
            for i in range(n):
                x, y = cx + random.uniform(-40, 40), cy + random.uniform(-40, 40)
                self.particles.append(Particle(x, y, COLORS[i % len(COLORS)]))
            self.global_best = (params["gx"], params["gy"], float('inf'))
            self.iteration = 0
            self.history = []
            self.history_list.delete(0, tk.END)
        self.running = True
        self._animate(params)

    def pause(self):
        self.running = False

    def reset(self):
        self.running = False
        self.particles, self.iteration, self.history = [], 0, []
        self.global_best = (0, 0, float('inf'))
        self.history_list.delete(0, tk.END)
        self.info.config(text="Iter: 0 | Fit: -")
        self.canvas.delete("all")

    def _animate(self, params):
        if not self.running: return
        if self.iteration >= params["max_iter"]:
            self.running = False
            messagebox.showinfo("Done", f"Best fitness: {self.global_best[2]:.4f}")
            return
        self.global_best = run_pso_step(self.particles, params["gx"], params["gy"],
                                       self.global_best, params["w"], params["c1"], params["c2"])
        self.iteration += 1
        snapshot = [(p.x, p.y, p.color) for p in self.particles]
        best_fit = self.global_best[2]
        self.history.append({"iter": self.iteration, "best_x": self.global_best[0],
                            "best_y": self.global_best[1], "fitness": best_fit, "positions": snapshot})
        self.history_list.insert(tk.END, f"Iter {self.iteration:3d} | Fit: {best_fit:7.2f}")
        self.history_list.see(tk.END)
        self._draw(params["gx"], params["gy"])
        self.info.config(text=f"Iter: {self.iteration} | Fit: {best_fit:.4f}")
        self.root.after(self.speed_var.get(), lambda: self._animate(params))

    def _draw(self, goal_x, goal_y):
        self.canvas.delete("all")
        for i in range(0, 650, 50):
            self.canvas.create_line(i, 0, i, 620, fill="#313244", dash=(2, 4))
            self.canvas.create_line(0, i, 620, i, fill="#313244", dash=(2, 4))
        r = 12
        self.canvas.create_oval(goal_x - r, goal_y - r, goal_x + r, goal_y + r,
                               fill="#f38ba8", outline="#fab387", width=2)
        self.canvas.create_text(goal_x, goal_y, text="★", fill="#1e1e2e", font=("Arial", 12, "bold"))
        for i, p in enumerate(self.particles):
            sz = 7
            self.canvas.create_oval(p.x - sz, p.y - sz, p.x + sz, p.y + sz,
                                   fill=p.color, outline="white", width=1)
            self.canvas.create_text(p.x, p.y, text=str(i + 1), fill="white", font=("Arial", 6, "bold"))
        gbx, gby = self.global_best[0], self.global_best[1]
        self.canvas.create_line(gbx - 8, gby, gbx + 8, gby, fill="#a6e3a1", width=2)
        self.canvas.create_line(gbx, gby - 8, gbx, gby + 8, fill="#a6e3a1", width=2)

    def show_selected(self):
        sel = self.history_list.curselection()
        if not sel: messagebox.showinfo("Select", "Select an iteration first"); return
        idx = sel[0]
        record = self.history[idx]
        params = self._get_params()
        if not params: return
        self.canvas.delete("all")
        for i in range(0, 651, 50):
            self.canvas.create_line(i, 0, i, 620, fill="#313244", dash=(2, 4))
            self.canvas.create_line(0, i, 620, i, fill="#313244", dash=(2, 4))
        gx, gy = params["gx"], params["gy"]
        r = 12
        self.canvas.create_oval(gx - r, gy - r, gx + r, gy + r, fill="#f38ba8", outline="#fab387", width=2)
        self.canvas.create_text(gx, gy, text="★", fill="#1e1e2e", font=("Arial", 12, "bold"))
        for i, (px, py, color) in enumerate(record["positions"]):
            sz = 7
            self.canvas.create_oval(px - sz, py - sz, px + sz, py + sz, fill=color, outline="white", width=1)
            self.canvas.create_text(px, py, text=str(i + 1), fill="white", font=("Arial", 6, "bold"))
        bx, by = record["best_x"], record["best_y"]
        self.canvas.create_line(bx - 8, by, bx + 8, by, fill="#a6e3a1", width=2)
        self.canvas.create_line(bx, by - 8, bx, by + 8, fill="#a6e3a1", width=2)
        self.canvas.create_text(310, 15, text=f"Iter {record['iter']} | Fit: {record['fitness']:.4f}",
                               fill="#89b4fa", font=("Arial", 10, "bold"))
