"""
Main GUI Application for Puzzle and Vacuum Solver
Modern GUI using tkinter with threading for background computation
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
from puzzle_solver import PuzzleSolver, PuzzleState
from vacuum_solver import VacuumSolver, VacuumState


class PuzzleVisualizer:
    """Visualize 8-puzzle game"""
    
    def __init__(self, canvas, cell_size=60):
        self.canvas = canvas
        self.cell_size = cell_size
        self.cell_padding = 5
    
    def draw_puzzle(self, state: PuzzleState):
        """Draw puzzle board"""
        self.canvas.delete("all")
        self.canvas.configure(bg='lightgray')
        
        for i in range(3):
            for j in range(3):
                x1 = j * self.cell_size + self.cell_padding
                y1 = i * self.cell_size + self.cell_padding
                x2 = x1 + self.cell_size - self.cell_padding
                y2 = y1 + self.cell_size - self.cell_padding
                
                value = state.board[i][j]
                
                if value == 0:
                    # Blank tile
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill='white', outline='black', width=2)
                else:
                    # Numbered tile
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill='#4CAF50', outline='black', width=2)
                    self.canvas.create_text((x1+x2)//2, (y1+y2)//2, text=str(value), 
                                          font=('Arial', 24, 'bold'), fill='white')


class VacuumVisualizer:
    """Visualize vacuum world"""
    
    def __init__(self, canvas, width=4, height=4, cell_size=60):
        self.canvas = canvas
        self.width = width
        self.height = height
        self.cell_size = cell_size
    
    def draw_vacuum(self, state: VacuumState):
        """Draw vacuum world"""
        self.canvas.delete("all")
        self.canvas.configure(bg='lightgray')
        
        for i in range(len(state.board)):
            for j in range(len(state.board[0])):
                x1 = j * self.cell_size
                y1 = i * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                
                cell = state.board[i][j]
                
                if (i, j) == state.vacuum_pos:
                    # Vacuum
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill='#FF9800', outline='black', width=2)
                    self.canvas.create_text((x1+x2)//2, (y1+y2)//2, text='V', 
                                          font=('Arial', 16, 'bold'), fill='white')
                elif cell == VacuumState.DIRT:
                    # Dirt
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill='#8B4513', outline='black', width=2)
                    self.canvas.create_text((x1+x2)//2, (y1+y2)//2, text='D', 
                                          font=('Arial', 16, 'bold'), fill='white')
                elif cell == VacuumState.OBSTACLE:
                    # Obstacle
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill='#333333', outline='black', width=2)
                    self.canvas.create_text((x1+x2)//2, (y1+y2)//2, text='#', 
                                          font=('Arial', 16, 'bold'), fill='white')
                else:
                    # Empty
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill='white', outline='black', width=1)


class SolverGUI:
    """Main GUI Application"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Search Algorithms Solver - 8-Puzzle & Vacuum World")
        self.root.geometry("1000x700")
        self.root.configure(bg='#f0f0f0')
        
        self.puzzle_solver = None
        self.vacuum_solver = None
        self.current_step = 0
        self.solution_steps = []
        self.is_solving = False

        self._configure_style()
        
        self._create_widgets()
        
        # Generate and display initial puzzles
        self._generate_new_puzzle()
        self._generate_new_vacuum()

    def _configure_style(self):
        """Configure a cleaner ttk visual style."""
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background="#f5f7fb")
        style.configure("TLabel", background="#f5f7fb", foreground="#1f2937")
        style.configure("TButton", padding=6)
        style.configure("TRadiobutton", background="#f5f7fb", foreground="#1f2937")
    
    def _create_widgets(self):
        """Create GUI widgets"""
        # Create notebook (tabs)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # 8-Puzzle Tab
        self.puzzle_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.puzzle_frame, text='8-Puzzle Solver')
        self._create_puzzle_tab()
        
        # Vacuum Tab
        self.vacuum_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.vacuum_frame, text='Vacuum World Solver')
        self._create_vacuum_tab()
    
    def _create_puzzle_tab(self):
        """Create 8-puzzle tab"""
        # Control panel
        control_frame = ttk.Frame(self.puzzle_frame)
        control_frame.pack(side='left', padx=10, pady=10)
        
        ttk.Label(control_frame, text="Algorithm:", font=('Arial', 10, 'bold')).pack()
        self.puzzle_algorithm = tk.StringVar(value="dfs")
        
        algorithms = [
            ("Depth-First Search", "dfs"),
            ("Breadth-First Search", "bfs"),
            ("Best-First Search (A*)", "best_fs")
        ]
        
        for text, value in algorithms:
            ttk.Radiobutton(control_frame, text=text, variable=self.puzzle_algorithm, 
                          value=value).pack(anchor='w')
        
        ttk.Separator(control_frame, orient='horizontal').pack(fill='x', pady=10)
        
        # Buttons
        self.puzzle_generate_btn = ttk.Button(control_frame, text="Generate New Puzzle", 
                                             command=self._generate_new_puzzle)
        self.puzzle_generate_btn.pack(fill='x', pady=5)
        
        self.puzzle_solve_btn = ttk.Button(control_frame, text="Solve", 
                                          command=self._solve_puzzle)
        self.puzzle_solve_btn.pack(fill='x', pady=5)
        
        self.puzzle_next_btn = ttk.Button(control_frame, text="Next Step", 
                                         command=self._puzzle_next_step, state='disabled')
        self.puzzle_next_btn.pack(fill='x', pady=5)
        
        self.puzzle_prev_btn = ttk.Button(control_frame, text="Previous Step", 
                                         command=self._puzzle_prev_step, state='disabled')
        self.puzzle_prev_btn.pack(fill='x', pady=5)
        
        self.puzzle_reset_btn = ttk.Button(control_frame, text="Reset", 
                                          command=self._puzzle_reset)
        self.puzzle_reset_btn.pack(fill='x', pady=5)
        
        self.puzzle_save_btn = ttk.Button(control_frame, text="Save Solution", 
                                         command=self._puzzle_save, state='disabled')
        self.puzzle_save_btn.pack(fill='x', pady=5)
        
        ttk.Separator(control_frame, orient='horizontal').pack(fill='x', pady=10)
        
        # Status label
        self.puzzle_status = ttk.Label(control_frame, text="Status: Ready", 
                                      font=('Arial', 9), foreground='blue')
        self.puzzle_status.pack()
        
        # Canvas for visualization
        canvas_frame = ttk.Frame(self.puzzle_frame)
        canvas_frame.pack(side='right', fill='both', expand=True, padx=10, pady=10)
        
        ttk.Label(canvas_frame, text="Board Visualization", font=('Arial', 12, 'bold')).pack()
        
        self.puzzle_canvas = tk.Canvas(canvas_frame, width=300, height=300, bg='lightgray')
        self.puzzle_canvas.pack(padx=10, pady=10)
        
        self.puzzle_visualizer = PuzzleVisualizer(self.puzzle_canvas)
        
        # Info panel
        info_frame = ttk.Frame(canvas_frame)
        info_frame.pack(fill='x', padx=10, pady=5)
        
        self.puzzle_info = ttk.Label(info_frame, text="", font=('Arial', 10))
        self.puzzle_info.pack()
    
    def _create_vacuum_tab(self):
        """Create vacuum world tab"""
        # Control panel
        control_frame = ttk.Frame(self.vacuum_frame)
        control_frame.pack(side='left', padx=10, pady=10)
        
        ttk.Label(control_frame, text="Algorithm:", font=('Arial', 10, 'bold')).pack()
        self.vacuum_algorithm = tk.StringVar(value="dfs")
        
        algorithms = [
            ("Depth-First Search", "dfs"),
            ("Breadth-First Search", "bfs"),
            ("Best-First Search", "best_fs")
        ]
        
        for text, value in algorithms:
            ttk.Radiobutton(control_frame, text=text, variable=self.vacuum_algorithm, 
                          value=value).pack(anchor='w')
        
        ttk.Separator(control_frame, orient='horizontal').pack(fill='x', pady=10)
        
        # Buttons
        self.vacuum_generate_btn = ttk.Button(control_frame, text="Generate New World", 
                                             command=self._generate_new_vacuum)
        self.vacuum_generate_btn.pack(fill='x', pady=5)
        
        self.vacuum_solve_btn = ttk.Button(control_frame, text="Solve", 
                                          command=self._solve_vacuum)
        self.vacuum_solve_btn.pack(fill='x', pady=5)
        
        self.vacuum_next_btn = ttk.Button(control_frame, text="Next Step", 
                                         command=self._vacuum_next_step, state='disabled')
        self.vacuum_next_btn.pack(fill='x', pady=5)
        
        self.vacuum_prev_btn = ttk.Button(control_frame, text="Previous Step", 
                                         command=self._vacuum_prev_step, state='disabled')
        self.vacuum_prev_btn.pack(fill='x', pady=5)
        
        self.vacuum_reset_btn = ttk.Button(control_frame, text="Reset", 
                                          command=self._vacuum_reset)
        self.vacuum_reset_btn.pack(fill='x', pady=5)
        
        self.vacuum_save_btn = ttk.Button(control_frame, text="Save Solution", 
                                         command=self._vacuum_save, state='disabled')
        self.vacuum_save_btn.pack(fill='x', pady=5)
        
        ttk.Separator(control_frame, orient='horizontal').pack(fill='x', pady=10)
        
        # Status label
        self.vacuum_status = ttk.Label(control_frame, text="Status: Ready", 
                                      font=('Arial', 9), foreground='blue')
        self.vacuum_status.pack()
        
        # Canvas for visualization
        canvas_frame = ttk.Frame(self.vacuum_frame)
        canvas_frame.pack(side='right', fill='both', expand=True, padx=10, pady=10)
        
        ttk.Label(canvas_frame, text="Board Visualization", font=('Arial', 12, 'bold')).pack()
        
        self.vacuum_canvas = tk.Canvas(canvas_frame, width=350, height=350, bg='lightgray')
        self.vacuum_canvas.pack(padx=10, pady=10)
        
        self.vacuum_visualizer = VacuumVisualizer(self.vacuum_canvas, 4, 4, 60)
        
        # Info panel
        info_frame = ttk.Frame(canvas_frame)
        info_frame.pack(fill='x', padx=10, pady=5)
        
        self.vacuum_info = ttk.Label(info_frame, text="", font=('Arial', 10))
        self.vacuum_info.pack()
    
    # ========== PUZZLE METHODS ==========
    
    def _generate_new_puzzle(self):
        """Generate and display a new random puzzle"""
        self.puzzle_solver = PuzzleSolver()
        self.puzzle_visualizer.draw_puzzle(self.puzzle_solver.initial_state)
        self.solution_steps = []
        self.current_step = 0
        self.puzzle_status.config(text="Status: Ready", foreground='blue')
        self.puzzle_next_btn.config(state='disabled')
        self.puzzle_prev_btn.config(state='disabled')
        self.puzzle_save_btn.config(state='disabled')
        self.puzzle_info.config(text="Click 'Solve' to find solution")
    
    def _solve_puzzle(self):
        """Solve puzzle in background thread"""
        if self.is_solving:
            messagebox.showwarning("Warning", "Already solving a puzzle!")
            return
        
        if not self.puzzle_solver:
            messagebox.showwarning("Warning", "Please generate a puzzle first!")
            return
        
        self.is_solving = True
        self.puzzle_solve_btn.config(state='disabled')
        self.puzzle_generate_btn.config(state='disabled')
        self.puzzle_status.config(text="Status: Solving...", foreground='orange')
        self.root.update()
        
        def solve():
            try:
                algorithm = self.puzzle_algorithm.get()
                
                if algorithm == "dfs":
                    success = self.puzzle_solver.solve_dfs()
                elif algorithm == "bfs":
                    success = self.puzzle_solver.solve_bfs()
                else:  # best_fs
                    success = self.puzzle_solver.solve_best_fs()
                
                if success:
                    self.solution_steps = self.puzzle_solver.get_solution_steps()
                    self.current_step = 0
                    self.root.after(0, self._update_puzzle_ui)
                else:
                    self.root.after(0, self._update_puzzle_no_solution_ui)
            except Exception as e:
                self.root.after(0, lambda: self._handle_puzzle_error(e))
        
        thread = threading.Thread(target=solve, daemon=True)
        thread.start()
    
    def _update_puzzle_ui(self):
        """Update puzzle UI after solving"""
        self.puzzle_visualizer.draw_puzzle(self.solution_steps[self.current_step])
        self.puzzle_status.config(text="Status: Solved!", foreground='green')
        self.puzzle_next_btn.config(state='normal')
        self.puzzle_prev_btn.config(state='normal')
        self.puzzle_save_btn.config(state='normal')
        self.puzzle_solve_btn.config(state='normal')
        self.puzzle_generate_btn.config(state='normal')
        self.is_solving = False
        # Required by assignment: write solution to solution.txt
        self.puzzle_solver.save_solution("solution.txt")
        self._update_puzzle_info()

    def _update_puzzle_no_solution_ui(self):
        """Update puzzle UI when no solution is found."""
        messagebox.showerror("Error", "No solution found!")
        self.puzzle_status.config(text="Status: No Solution", foreground='red')
        self.puzzle_solve_btn.config(state='normal')
        self.puzzle_generate_btn.config(state='normal')
        self.is_solving = False

    def _handle_puzzle_error(self, error):
        """Handle puzzle solver errors on UI thread."""
        messagebox.showerror("Error", str(error))
        self.puzzle_solve_btn.config(state='normal')
        self.puzzle_generate_btn.config(state='normal')
        self.is_solving = False
    
    def _puzzle_next_step(self):
        """Show next puzzle step"""
        if self.current_step < len(self.solution_steps) - 1:
            self.current_step += 1
            self.puzzle_visualizer.draw_puzzle(self.solution_steps[self.current_step])
            self._update_puzzle_info()
    
    def _puzzle_prev_step(self):
        """Show previous puzzle step"""
        if self.current_step > 0:
            self.current_step -= 1
            self.puzzle_visualizer.draw_puzzle(self.solution_steps[self.current_step])
            self._update_puzzle_info()
    
    def _puzzle_reset(self):
        """Reset puzzle"""
        self.solution_steps = []
        self.current_step = 0
        if self.puzzle_solver:
            self.puzzle_visualizer.draw_puzzle(self.puzzle_solver.initial_state)
        else:
            self.puzzle_canvas.delete("all")
        self.puzzle_status.config(text="Status: Ready", foreground='blue')
        self.puzzle_next_btn.config(state='disabled')
        self.puzzle_prev_btn.config(state='disabled')
        self.puzzle_save_btn.config(state='disabled')
        self.puzzle_info.config(text="Click 'Solve' to find solution" if self.puzzle_solver else "")
    
    def _puzzle_save(self):
        """Save puzzle solution"""
        if self.puzzle_solver:
            self.puzzle_solver.save_solution("solution.txt")
            msg = f"Saved to solution.txt\n\nAlgorithm: {self.puzzle_solver.algorithm_name}\nSteps: {len(self.solution_steps)-1}\nTotal Cost: {self.puzzle_solver.total_cost}"
            messagebox.showinfo("Saved", msg)
            info = f"Algorithm: {self.puzzle_solver.algorithm_name}\n"
            info += f"Step: {self.current_step}/{len(self.solution_steps) - 1}\n"
            info += f"Total Cost: {self.puzzle_solver.total_cost}\n"
            info += f"Total Steps: {len(self.solution_steps) - 1}"
            self.puzzle_info.config(text=info)
    
    def _update_puzzle_info(self):
        """Update puzzle info labels"""
        if self.puzzle_solver and len(self.solution_steps) > 0:
            info = f"Algorithm: {self.puzzle_solver.algorithm_name}\n"
            info += f"Step: {self.current_step}/{len(self.solution_steps) - 1}\n"
            info += f"Total Cost: {self.puzzle_solver.total_cost}\n"
            info += f"Total Steps: {len(self.solution_steps) - 1}"
            self.puzzle_info.config(text=info)
    
    # ========== VACUUM METHODS ==========
    
    def _generate_new_vacuum(self):
        """Generate and display a new random vacuum world"""
        self.vacuum_solver = VacuumSolver(width=4, height=4, num_obstacles=3)
        self.vacuum_visualizer.draw_vacuum(self.vacuum_solver.initial_state)
        self.solution_steps = []
        self.current_step = 0
        self.vacuum_status.config(text="Status: Ready", foreground='blue')
        self.vacuum_next_btn.config(state='disabled')
        self.vacuum_prev_btn.config(state='disabled')
        self.vacuum_save_btn.config(state='disabled')
        self.vacuum_info.config(text="Click 'Solve' to find solution")
    
    def _solve_vacuum(self):
        """Solve vacuum in background thread"""
        if self.is_solving:
            messagebox.showwarning("Warning", "Already solving!")
            return
        
        if not self.vacuum_solver:
            messagebox.showwarning("Warning", "Please generate a world first!")
            return
        
        self.is_solving = True
        self.vacuum_solve_btn.config(state='disabled')
        self.vacuum_generate_btn.config(state='disabled')
        self.vacuum_status.config(text="Status: Solving...", foreground='orange')
        self.root.update()
        
        def solve():
            try:
                algorithm = self.vacuum_algorithm.get()
                
                if algorithm == "dfs":
                    self.vacuum_solver.solve_dfs()
                elif algorithm == "bfs":
                    self.vacuum_solver.solve_bfs()
                else:  # best_fs
                    self.vacuum_solver.solve_best_fs()
                
                if self.vacuum_solver.has_solution:
                    self.solution_steps = self.vacuum_solver.get_solution_steps()
                    self.current_step = 0
                    self.root.after(0, self._update_vacuum_ui)
                else:
                    self.solution_steps = [self.vacuum_solver.initial_state]
                    self.root.after(0, self._update_vacuum_ui_no_solution)
            except Exception as e:
                self.root.after(0, lambda: self._handle_vacuum_error(e))
        
        thread = threading.Thread(target=solve, daemon=True)
        thread.start()
    
    def _update_vacuum_ui(self):
        """Update vacuum UI after solving"""
        self.vacuum_visualizer.draw_vacuum(self.solution_steps[self.current_step])
        self.vacuum_status.config(text="Status: Solved!", foreground='green')
        self.vacuum_next_btn.config(state='normal')
        self.vacuum_prev_btn.config(state='normal')
        self.vacuum_save_btn.config(state='normal')
        self.vacuum_solve_btn.config(state='normal')
        self.vacuum_generate_btn.config(state='normal')
        self.is_solving = False
        # Required by assignment: write solution to solution.txt
        self.vacuum_solver.save_solution("solution.txt")
        self._update_vacuum_info()
    
    def _update_vacuum_ui_no_solution(self):
        """Update vacuum UI when no solution"""
        self.vacuum_visualizer.draw_vacuum(self.solution_steps[0])
        self.vacuum_status.config(text="Status: No Solution!", foreground='red')
        self.vacuum_save_btn.config(state='normal')
        self.vacuum_solve_btn.config(state='normal')
        self.vacuum_generate_btn.config(state='normal')
        self.is_solving = False
        self.vacuum_solver.save_solution("solution.txt")
        self.vacuum_info.config(text="No solution: Obstacles block path to dirt")

    def _handle_vacuum_error(self, error):
        """Handle vacuum solver errors on UI thread."""
        messagebox.showerror("Error", str(error))
        self.vacuum_solve_btn.config(state='normal')
        self.vacuum_generate_btn.config(state='normal')
        self.is_solving = False
    
    def _vacuum_next_step(self):
        """Show next vacuum step"""
        if self.current_step < len(self.solution_steps) - 1:
            self.current_step += 1
            self.vacuum_visualizer.draw_vacuum(self.solution_steps[self.current_step])
            self._update_vacuum_info()
    
    def _vacuum_prev_step(self):
        """Show previous vacuum step"""
        if self.current_step > 0:
            self.current_step -= 1
            self.vacuum_visualizer.draw_vacuum(self.solution_steps[self.current_step])
            self._update_vacuum_info()
    
    def _vacuum_reset(self):
        """Reset vacuum"""
        self.solution_steps = []
        self.current_step = 0
        if self.vacuum_solver:
            self.vacuum_visualizer.draw_vacuum(self.vacuum_solver.initial_state)
        else:
            self.vacuum_canvas.delete("all")
        self.vacuum_status.config(text="Status: Ready", foreground='blue')
        self.vacuum_next_btn.config(state='disabled')
        self.vacuum_prev_btn.config(state='disabled')
        self.vacuum_save_btn.config(state='disabled')
        self.vacuum_info.config(text="Click 'Solve' to find solution" if self.vacuum_solver else "")
    
    def _vacuum_save(self):
        """Save vacuum solution"""
        if self.vacuum_solver:
            self.vacuum_solver.save_solution("solution.txt")
            if self.vacuum_solver.has_solution:
                msg = f"Saved to solution.txt\n\nAlgorithm: {self.vacuum_solver.algorithm_name}\nSteps: {len(self.solution_steps)-1}\nTotal Cost: {self.vacuum_solver.total_cost}"
            else:
                msg = "Saved to solution.txt\n\nNo solution - blocked by obstacles."
            messagebox.showinfo("Saved", msg)
            info = f"Algorithm: {self.vacuum_solver.algorithm_name}\n"
            info += f"Step: {self.current_step}/{len(self.solution_steps) - 1}\n"
            info += f"Total Cost: {self.vacuum_solver.total_cost}\n"
            info += f"Total Steps: {len(self.solution_steps) - 1}"
            self.vacuum_info.config(text=info)
    
    def _update_vacuum_info(self):
        """Update vacuum info labels"""
        if self.vacuum_solver:
            if self.vacuum_solver.has_solution and len(self.solution_steps) > 0:
                info = f"Algorithm: {self.vacuum_solver.algorithm_name}\n"
                info += f"Step: {self.current_step}/{len(self.solution_steps) - 1}\n"
                info += f"Total Cost: {self.vacuum_solver.total_cost}\n"
                info += f"Total Steps: {len(self.solution_steps) - 1}"
                self.vacuum_info.config(text=info)
            elif not self.vacuum_solver.has_solution:
                self.vacuum_info.config(text="No solution: Obstacles block path")


def main():
    """Run the application"""
    root = tk.Tk()
    app = SolverGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
