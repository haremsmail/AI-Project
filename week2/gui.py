import tkinter as tk
from tkinter import messagebox, font
import threading
""" runing task in backgroun"""
from board import Board
from vacuum_dfs import VacuumDFS
"""" wata import dipth first serach alogoithm"""
from solution_writer import SolutionWriter
"""" au file writrer stepakan pishan dadan"""


class VacuumGUI:
    """ this is type of color hexa"""
    COLORS = {
        
        'bg': '#0a1628', 'bg_light': '#1a2f5a', 'primary': '#00d4ff',
        'success': '#00ff88', 'danger': '#ff3366', 'warning': '#ffaa00',
        'accent': '#ff00ff', 'vacuum': '#3366ff', 'dirt': '#00dd88',
        'obstacle': '#333333', 'empty': '#f5f5f5', 'path': '#ffaa00'
    }
    
    def __init__(self, root):
        """ the root means main window"""
        self.root = root
        self.root.title("Vacuum Cleaner - DFS Solver")
        self.root.geometry("1200x750")
        """ this is a set of window size"""
        self.root.configure(bg=self.COLORS['bg'])
        """ this is used to set background color of the window"""
        """ used to for background"""
        """ no bord yet no solver yeat"""
        self.board = self.solver = None
        self.solution_path = []
        """ store solution steps  au solution path baakar de bo auayduatre stepakan lanau au fila xazn bn """
        self.current_step = self.total_cost = 0
        """" wata agar dast pe dakay au duana hardukayan sfra"""
        self.solving_in_progress = False
        """ agar true bu solve daka agar false bu dasty pe nakrdua"""
        
        self.setup_ui()
        """ call function to create interface lo drustk krndy button gird  """
    
    def setup_ui(self):
        main = tk.Frame(self.root, bg=self.COLORS['bg'])
        """ orgianize button all thisng group things togoerhter"""
        main.pack(fill=tk.BOTH, expand=True)
        """ expand wata fround krdna  tell tkinter haw dispaly"""
        """ pack main width haeight  expand available space"""
        
        # Title
        title = tk.Label(main, text="VACUUM CLEANER", font=("Arial", 28, "bold"),
                        bg=self.COLORS['bg'], fg=self.COLORS['primary'])
        title.pack(pady=15)
        """ auayan bo zyatkrndy space above and below title"""
        
        # Content
        content = tk.Frame(main, bg=self.COLORS['bg'])
        content.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        """ au du line conteriank drust daka
        yakakayan bo contorloer yakayan bo game"""
        
        # Left: Board
        left = tk.Frame(content, bg=self.COLORS['bg'])
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        board_container = tk.Frame(left, bg=self.COLORS['bg_light'], relief=tk.RAISED, bd=3)
        board_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        """ addding margina and space
        bd=3 → border thickness = 3 pixels wata astury stun"""
        
        self.canvas = tk.Canvas(board_container, bg=self.COLORS['empty'],
                               width=420, height=420, highlightthickness=0)
        """ au drezhe remove border aoutside and canvas drowing hamu shtakan da"""
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        """ canvaas rubary wenakeshanaa lanau border cotnaisner drust dabe"""
        
        # Right: Controls
        right = tk.Frame(content, bg=self.COLORS['bg'], width=280)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, padx=10)
        right.pack_propagate(False)
        """ mahela prograta sizaka bgore agar gawratrysh u"""
        
        # Stats create 3 ui seciton
        for label, ref in [("COST", "cost"), ("STATUS", "status"), ("MOVES", "moves")]:
            """ wata loop list of tuple har yakayan manay lable ref"""
            tk.Label(right, text=label, font=("Arial", 8), bg=self.COLORS['bg'],
                    fg='white').pack(pady=(10, 2))
            stat = tk.Label(right, text="0" if ref != "status" else "READY",
                           font=("Arial", 16, "bold"), bg=self.COLORS['bg'],
                           fg=self.COLORS[['danger', 'accent', 'warning'][['cost', 'status', 'moves'].index(ref)]])
            stat.pack(pady=(0, 10))
            setattr(self, f'{ref}_label', stat)
            """ this is used to save label create variable name dynamically  cost_label status_label moves_label"""
        
        # Buttons
        btn_frame = tk.Frame(right, bg=self.COLORS['bg'])
        btn_frame.pack(fill=tk.X)
        """wata tanya ba arasty x kshany haya"""
        
        buttons = [("⟳ GENERATE", self.generate_board, self.COLORS['primary']),
                  ("⚡ SOLVE", self.solve, self.COLORS['success']),
                  ("🔀 SHUFFLE", self.generate_board, self.COLORS['warning']),
                  ("◀ PREV", self.previous_step, self.COLORS['danger']),
                  ("NEXT ▶", self.next_step, self.COLORS['accent']),
                  ("↻ RESET", self.reset_solution, self.COLORS['warning'])]
        
        for txt, cmd, col in buttons:
            tk.Button(btn_frame, text=txt, command=cmd, bg=col, fg='#000', 
                     font=("Arial", 10, "bold"), width=17, relief=tk.FLAT).pack(pady=4)
            """ button drust daka ba text command lo ruanakrdyn esh krdy funciton """
        
       
       
        # Info Panel
        """ box panel drust daka info"""
        info = tk.Frame(right, bg=self.COLORS['bg_light'], relief=tk.RAISED, bd=2)
        info.pack(fill=tk.X, pady=(15, 0))
        """ create box panel for information"""



        self.step_label = tk.Label(info, text="Step: 0/0", font=("Arial", 10, "bold"),
                                  bg=self.COLORS['bg_light'], fg=self.COLORS['primary'])
        self.step_label.pack(pady=5)
        
        self.move_label = tk.Label(info, text="Ready", font=("Arial", 9),
                                  bg=self.COLORS['bg_light'], fg=self.COLORS['accent'])
        self.move_label.pack(pady=5)
        
        # Legend explianed of symoble
        tk.Label(info, text="LEGEND", font=("Arial", 10, "bold"),
                bg=self.COLORS['bg_light'], fg=self.COLORS['primary']).pack(pady=(10, 5))
        
        legend_items = [("V", self.COLORS['vacuum'], "Vacuum"),
                       ("G", self.COLORS['dirt'], "Goal"),
                       ("#", self.COLORS['obstacle'], "Obstacle"),
                       ("●", self.COLORS['path'], "Current")]
        
        for sym, col, txt in legend_items:
            
            row = tk.Frame(info, bg=self.COLORS['bg_light'])
            row.pack(fill=tk.X, padx=5, pady=2)
            tk.Label(row, text=sym, font=("Arial", 11, "bold"), bg=col,
                    fg='white', width=3).pack(side=tk.LEFT, padx=3)
            tk.Label(row, text=txt, font=("Arial", 8), bg=self.COLORS['bg_light'],
                    fg='white').pack(side=tk.LEFT)
        
        self.root.bind("<Configure>", self._on_resize)
    
    def _on_resize(self, event):
        """ bordaka dakeshetau katek qabara panjara dagore"""
        if self.board:
            self.draw_board(self.current_step if self.current_step > 0 else None)
            """ drow speicf stepakan"""
    
    def generate_board(self):
        """ katek click la genearet new borde"""
        self.board = Board(width=6, height=6)
        self.board.generate_random(obstacle_count=8)
        """ by default 6 dandaraua"""
        self.solution_path = []
        self.current_step = self.total_cost = 0
        self.solver = None
        
        self.status_label.config(text="Ready")
        self.cost_label.config(text="0")
        self.moves_label.config(text="0")
        self.step_label.config(text="0/0")
        self.move_label.config(text="Ready")
        self.draw_board()
    """ lera ra auastm"""
    def solve(self):
        """" statrt solving process dfs and updated ui"""
        if not self.board:
            messagebox.showwarning("Warning", "Generate board first!")
            return
        if self.solving_in_progress:
            """" regry daka 2 solve pekaua ru bdan"""
            return
        
        self.solving_in_progress = True
        self.status_label.config(text="SOLVING", fg=self.COLORS['warning'])
        self.move_label.config(text="Finding solution...")
        
        thread = threading.Thread(target=self._solve_bg)
        """ wak krekareky jia waya ka codaka bareua daba au threada chand shtek pekaua esh ka ui bgore solving  dast pe bkda
        lagal depth first search"""
        thread.daemon = True
        """ if program stop thread stop dfs esh daka be auay backgroufnd free krdn"""
        thread.start()
    
    def _solve_bg(self):
        """ run depth first serach in backgorund"""
        self.solver = VacuumDFS(self.board)
        """ create depth first serach of current node"""
        success, path, cost, msg = self.solver.solve()
        """ is realy importan call algorithm"""
        
        if success:
            """ if found depth firt search in thi path """
            self.solution_path = self.solver.path
            self.total_cost = cost
            self.root.after(0, self._update_solve, True, len(path))
            """ this is used to thread to updated gui"""
        else:
            self.root.after(0, self._update_solve, False, 0)
            """ solution not found"""
    


    def _update_solve(self, success, count):
        """ updated gui after solving finshed"""
        self.solving_in_progress = False
        
        if success:
            """ it means if soluction foudn"""
            self.status_label.config(text="SOLVED!", fg=self.COLORS['success'])
            self.cost_label.config(text=str(self.total_cost))
            self.moves_label.config(text=str(count))
            """ dispaly number of stepm """
            self.step_label.config(text=f"0/{count}")
            self.move_label.config(text="Click NEXT to view")
            writer = SolutionWriter()
            writer.write(self.board, self.solution_path, self.total_cost, True, self.solver)
            """"Save:

board
path
cost
success = True
solver details"""
        else:
            self.status_label.config(text="NO SOLUTION", fg=self.COLORS['danger'])
            self.move_label.config(text="Obstacles block path")
            """ obstecalse is blocked"""
            writer = SolutionWriter()
            writer.write(self.board, [], 0, False, self.solver)
        
        self.draw_board()
        """ upated display"""
    
    def next_step(self):
        """ move the next step and updated display"""
        if self.solution_path and self.current_step < len(self.solution_path):
            self.current_step += 1
            self.update_display()
    
    def previous_step(self):
        if self.current_step > 0:
            self.current_step -= 1
            self.update_display()
    
    def reset_solution(self):
        if self.solution_path:
            self.current_step = 0
            self.update_display()
    
    def update_display(self):
        """"Updates the UI to show the current step, move, and redraw the board"""
        total = len(self.solution_path)
        self.step_label.config(text=f"Step: {self.current_step}/{total}")
        """ updated stepakan"""
        
        if self.current_step == 0:
            self.move_label.config(text="Start")
        elif self.current_step <= total:
            move = self.solution_path[self.current_step - 1]
            """ list start and index 0 but step start 1 so -1"""
            self.move_label.config(text=f"{move['move']} (Cost: {move['cost']})")
        

        self.draw_board(self.current_step if self.current_step > 0 else None)
        """ drow reord """
    
    def draw_board(self, highlight=None):
        """ hilight stepakan ka rud dadan pishan dad"""
        self.canvas.delete("all")
        """ clear canvas and shtakan hazr daka lo updated"""
        if not self.board:
            """ agar borad nabu retunrny hih naka"""
            return
        
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        """get canvas size calcualt size canvas"""
        size = (min(cw, ch) - 20) // 6 if min(cw, ch) > 100 else 70
        """ divide  6 sel
        canvas = 420 × 420

Then:

size = (420 - 20) // 6 = 66"""
        
        colors = {Board.OBSTACLE: self.COLORS['obstacle'], Board.DIRT: self.COLORS['dirt'],
                 Board.VACUUM: self.COLORS['vacuum'], Board.EMPTY: self.COLORS['empty']}
        """ au coda torakau har xanayak lasar canvaka dakeshet esh lasar hamu sht daka"""
        for r in range(6):
            for c in range(6):
                """ means row and colum"""
                """ get what is inside in each s"""
                x1, y1 = c * size, r * size
                cell = self.board.grid[r][c]
          
                
                color = colors.get(cell, self.COLORS['empty'])
                """ wata step haya bo auay tishky baxna """
                if highlight and 0 < highlight <= len(self.solution_path):
                    tr, tc = self.solution_path[highlight - 1]['to']
                    """ danany shuenaka bo aua stepay boy aroy"""
                    if r == tr and c == tc:
                        """ wata au sheuany leyty current postion bu lagal stepaka"""
                        color = self.COLORS['path']
                        """ xanakan dapshekan rectauerl agar current step bu lagal ahar yakan zyad daka """
                
                self.canvas.create_rectangle(x1, y1, x1 + size, y1 + size,
                                            fill=color, outline='#ccc', width=1)
                
                text = {Board.OBSTACLE: "#", Board.DIRT: "G", Board.VACUUM: "V"}.get(cell, "")
                if text:
                    self.canvas.create_text(x1 + size//2, y1 + size//2, text=text,
                                           font=("Arial", 14, "bold"), fill='white')


def main():
    root = tk.Tk()
    gui = VacuumGUI(root)
    root.mainloop()
    """ keeep the windoe runing"""


if __name__ == "__main__":
    main()
