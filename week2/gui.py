"""
Modern Interactive GUI for Vacuum Cleaner DFS Solver
Professional Tkinter Interface with Beautiful Design
"""
import tkinter as tk
from tkinter import messagebox, font
import threading
from board import Board
from vacuum_dfs import VacuumDFS
from solution_writer import SolutionWriter


class VacuumGUI:
    """Modern Interactive GUI for Vacuum Cleaner Problem"""
    
    # Modern Color Scheme
    COLORS = {
        'bg_dark': '#0a1628',        # Dark blue background
        'bg_light': '#1a2f5a',       # Lighter blue
        'primary': '#00d4ff',        # Cyan
        'success': '#00ff88',        # Green
        'danger': '#ff3366',         # Red/Pink
        'warning': '#ffaa00',        # Orange
        'accent1': '#ff00ff',        # Magenta
        'accent2': '#00ffff',        # Bright Cyan
        
        'vacuum': '#3366ff',         # Blue
        'dirt': '#00dd88',           # Green
        'obstacle': '#333333',       # Dark
        'empty': '#f5f5f5',          # Light gray
        'path': '#ffaa00',           # Orange
        'white': '#ffffff',
    }
    
    def __init__(self, root):
        self.root = root
        self.root.title("Vacuum Cleaner - DFS Solver")
        self.root.geometry("1200x800")
        self.root.configure(bg=self.COLORS['bg_dark'])
        self.root.resizable(True, True)  # Allow maximize
        
        self.board = None
        self.solver = None
        self.solution_path = []
        self.current_step = 0
        self.total_cost = 0
        self.solving_in_progress = False
        
        # Manual play variables
        self.manual_play_enabled = False
        self.manual_vacuum_pos = None
        self.manual_cost = 0
        self.manual_mode_label = None
        self.manual_solution_path = []  # Track manual moves like solution path
        
        # Setup UI
        self.setup_ui()
        
        # Bind keyboard events for manual play
        self.root.bind('<Up>', self.on_key_press)
        self.root.bind('<Down>', self.on_key_press)
        self.root.bind('<Left>', self.on_key_press)
        self.root.bind('<Right>', self.on_key_press)
        self.root.bind('<w>', self.on_key_press)
        self.root.bind('<W>', self.on_key_press)
        self.root.bind('<s>', self.on_key_press)
        self.root.bind('<S>', self.on_key_press)
        self.root.bind('<a>', self.on_key_press)
        self.root.bind('<A>', self.on_key_press)
        self.root.bind('<d>', self.on_key_press)
        self.root.bind('<D>', self.on_key_press)
    
    def setup_ui(self):
        """Setup the modern professional user interface"""
        # Main container with centered layout
        main_frame = tk.Frame(self.root, bg=self.COLORS['bg_dark'])
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title section
        self.setup_title(main_frame)
        
        # Board and buttons in one row
        content_frame = tk.Frame(main_frame, bg=self.COLORS['bg_dark'])
        content_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=10)
        
        # Left: Board section (expands)
        self.setup_board_section(content_frame)
        
        # Right: Buttons and info (fixed width)
        self.setup_right_panel(content_frame)
    
    def setup_title(self, parent):
        """Setup title section"""
        title_frame = tk.Frame(parent, bg=self.COLORS['bg_dark'])
        title_frame.pack(pady=(30, 10))
        
        title_font = font.Font(family="Arial", size=36, weight="bold")
        tk.Label(title_frame, text="VACUUM CLEANER",
                font=title_font, bg=self.COLORS['bg_dark'],
                fg=self.COLORS['primary']).pack()
        
        subtitle_font = font.Font(family="Arial", size=12)
        tk.Label(title_frame, text="INTELLIGENT SEARCH ENGINE",
                font=subtitle_font, bg=self.COLORS['bg_dark'],
                fg=self.COLORS['accent2']).pack()
    
    def setup_stats(self, parent):
        """Setup statistics display (moved to right panel)"""
        pass
    
    def setup_right_panel(self, parent):
        """Setup right panel with buttons and info"""
        right_frame = tk.Frame(parent, bg=self.COLORS['bg_dark'], width=280)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, padx=10)
        right_frame.pack_propagate(False)
        
        # Stats at top
        stat_frame = tk.Frame(right_frame, bg=self.COLORS['bg_dark'])
        stat_frame.pack(pady=(0, 15))
        
        self.moves_stat_label = self.create_stat_box(stat_frame, "MOVES", "0", self.COLORS['warning'])
        self.moves_stat_label.pack(pady=5)
        
        self.cost_stat_label = self.create_stat_box(stat_frame, "COST", "0", self.COLORS['danger'])
        self.cost_stat_label.pack(pady=5)
        
        self.status_stat_label = self.create_stat_box(stat_frame, "STATUS", "READY", self.COLORS['accent2'])
        self.status_stat_label.pack(pady=5)
        
        # Buttons
        buttons_frame = tk.Frame(right_frame, bg=self.COLORS['bg_dark'])
        buttons_frame.pack(fill=tk.BOTH, expand=True)
        
        self.btn_generate = self.create_modern_button(
            buttons_frame, "⟳ GENERATE", self.generate_board, 
            self.COLORS['primary'])
        self.btn_generate.pack(fill=tk.X, pady=6)
        
        self.btn_solve = self.create_modern_button(
            buttons_frame, "⚡ SOLVE", self.solve,
            self.COLORS['success'])
        self.btn_solve.pack(fill=tk.X, pady=6)
        
        self.btn_shuffle = self.create_modern_button(
            buttons_frame, "🔀 SHUFFLE", self.shuffle_board,
            self.COLORS['warning'])
        self.btn_shuffle.pack(fill=tk.X, pady=6)
        
        self.btn_play = self.create_modern_button(
            buttons_frame, "🎮 PLAY MANUALLY", self.start_manual_play,
            self.COLORS['accent1'])
        self.btn_play.pack(fill=tk.X, pady=6)
        
        tk.Frame(right_frame, bg=self.COLORS['bg_dark']).pack(pady=8)
        
        self.btn_previous = self.create_modern_button(
            buttons_frame, "◀ PREVIOUS", self.previous_step,
            self.COLORS['danger'])
        self.btn_previous.pack(fill=tk.X, pady=6)
        
        self.btn_next = self.create_modern_button(
            buttons_frame, "NEXT ▶", self.next_step,
            self.COLORS['accent1'])
        self.btn_next.pack(fill=tk.X, pady=6)
        
        self.btn_reset = self.create_modern_button(
            buttons_frame, "↻ RESET", self.reset_solution,
            self.COLORS['warning'])
        self.btn_reset.pack(fill=tk.X, pady=6)
        
        # Step and detail info
        info_frame = tk.Frame(right_frame, bg=self.COLORS['bg_light'], 
                             relief=tk.RAISED, bd=2)
        info_frame.pack(fill=tk.X, pady=(10, 0))
        
        step_font = font.Font(family="Arial", size=11, weight="bold")
        self.step_label = tk.Label(info_frame, text="Step: 0/0",
                                  font=step_font,
                                  bg=self.COLORS['bg_light'],
                                  fg=self.COLORS['primary'])
        self.step_label.pack(pady=5)
        
        detail_font = font.Font(family="Arial", size=9)
        self.detail_label = tk.Label(info_frame, text="Ready to start",
                                    font=detail_font,
                                    bg=self.COLORS['bg_light'],
                                    fg=self.COLORS['accent2'])
        self.detail_label.pack(pady=5)
        
        # Manual mode indicator
        mode_font = font.Font(family="Arial", size=8, weight="bold")
        self.manual_mode_label = tk.Label(info_frame, text="AUTO MODE - Click PLAY to manual",
                                         font=mode_font,
                                         bg=self.COLORS['bg_light'],
                                         fg=self.COLORS['primary'])
        self.manual_mode_label.pack(pady=5)
        
        # LEGEND - Explain symbols
        legend_title = tk.Label(info_frame, text="📖 LEGEND",
                               font=("Arial", 11, "bold"),
                               bg=self.COLORS['bg_light'],
                               fg=self.COLORS['primary'])
        legend_title.pack(pady=(10, 8))
        
        # V = Vacuum Robot
        v_frame = tk.Frame(info_frame, bg=self.COLORS['bg_light'])
        v_frame.pack(fill=tk.X, padx=5, pady=(5, 0))
        v_box = tk.Label(v_frame, text="V", font=("Arial", 14, "bold"),
                        bg=self.COLORS['vacuum'], fg='white', width=3)
        v_box.pack(side=tk.LEFT, padx=5, pady=3)
        v_text_frame = tk.Frame(info_frame, bg=self.COLORS['bg_light'])
        v_text_frame.pack(fill=tk.X, padx=12, pady=(0, 5))
        tk.Label(v_text_frame, text="VACUUM (Robot Cleaner)", 
                font=("Arial", 8, "bold"), bg=self.COLORS['bg_light'], fg=self.COLORS['primary']).pack(anchor=tk.W)
        tk.Label(v_text_frame, text="The cleaning robot that moves on the board", 
                font=("Arial", 7), bg=self.COLORS['bg_light'], fg='#aaaaaa').pack(anchor=tk.W)
        tk.Label(v_text_frame, text="- Starts at initial position (blue square)", 
                font=("Arial", 7), bg=self.COLORS['bg_light'], fg='#999999').pack(anchor=tk.W)
        tk.Label(v_text_frame, text="- Can move UP, DOWN, LEFT, RIGHT one square", 
                font=("Arial", 7), bg=self.COLORS['bg_light'], fg='#999999').pack(anchor=tk.W)
        
        # G = Dirt/Goal
        g_frame = tk.Frame(info_frame, bg=self.COLORS['bg_light'])
        g_frame.pack(fill=tk.X, padx=5, pady=(5, 0))
        g_box = tk.Label(g_frame, text="G", font=("Arial", 14, "bold"),
                        bg=self.COLORS['dirt'], fg='white', width=3)
        g_box.pack(side=tk.LEFT, padx=5, pady=3)
        g_text_frame = tk.Frame(info_frame, bg=self.COLORS['bg_light'])
        g_text_frame.pack(fill=tk.X, padx=12, pady=(0, 5))
        tk.Label(g_text_frame, text="GOAL / DIRT", 
                font=("Arial", 8, "bold"), bg=self.COLORS['bg_light'], fg=self.COLORS['success']).pack(anchor=tk.W)
        tk.Label(g_text_frame, text="The target destination where vacuum must reach", 
                font=("Arial", 7), bg=self.COLORS['bg_light'], fg='#aaaaaa').pack(anchor=tk.W)
        tk.Label(g_text_frame, text="- The objective of the puzzle", 
                font=("Arial", 7), bg=self.COLORS['bg_light'], fg='#999999').pack(anchor=tk.W)
        tk.Label(g_text_frame, text="- Must find optimal path to reach it", 
                font=("Arial", 7), bg=self.COLORS['bg_light'], fg='#999999').pack(anchor=tk.W)
        
        # # = Obstacle
        hash_frame = tk.Frame(info_frame, bg=self.COLORS['bg_light'])
        hash_frame.pack(fill=tk.X, padx=5, pady=(5, 0))
        hash_box = tk.Label(hash_frame, text="#", font=("Arial", 14, "bold"),
                           bg=self.COLORS['obstacle'], fg='white', width=3)
        hash_box.pack(side=tk.LEFT, padx=5, pady=3)
        hash_text_frame = tk.Frame(info_frame, bg=self.COLORS['bg_light'])
        hash_text_frame.pack(fill=tk.X, padx=12, pady=(0, 5))
        tk.Label(hash_text_frame, text="OBSTACLE / WALL", 
                font=("Arial", 8, "bold"), bg=self.COLORS['bg_light'], fg=self.COLORS['danger']).pack(anchor=tk.W)
        tk.Label(hash_text_frame, text="Blocks the path - vacuum cannot pass through", 
                font=("Arial", 7), bg=self.COLORS['bg_light'], fg='#aaaaaa').pack(anchor=tk.W)
        tk.Label(hash_text_frame, text="- Creates challenging puzzle scenarios", 
                font=("Arial", 7), bg=self.COLORS['bg_light'], fg='#999999').pack(anchor=tk.W)
        tk.Label(hash_text_frame, text="- May make goal unreachable", 
                font=("Arial", 7), bg=self.COLORS['bg_light'], fg='#999999').pack(anchor=tk.W)
        
        # Orange = Current Path Step
        path_frame = tk.Frame(info_frame, bg=self.COLORS['bg_light'])
        path_frame.pack(fill=tk.X, padx=5, pady=(5, 0))
        path_box = tk.Label(path_frame, text="●", font=("Arial", 14, "bold"),
                           bg=self.COLORS['path'], fg='white', width=3)
        path_box.pack(side=tk.LEFT, padx=5, pady=3)
        path_text_frame = tk.Frame(info_frame, bg=self.COLORS['bg_light'])
        path_text_frame.pack(fill=tk.X, padx=12, pady=(0, 5))
        tk.Label(path_text_frame, text="CURRENT STEP", 
                font=("Arial", 8, "bold"), bg=self.COLORS['bg_light'], fg=self.COLORS['warning']).pack(anchor=tk.W)
        tk.Label(path_text_frame, text="Highlighted position during solution playback", 
                font=("Arial", 7), bg=self.COLORS['bg_light'], fg='#aaaaaa').pack(anchor=tk.W)
        tk.Label(path_text_frame, text="- Shows where vacuum is at each step", 
                font=("Arial", 7), bg=self.COLORS['bg_light'], fg='#999999').pack(anchor=tk.W)
        tk.Label(path_text_frame, text="- Navigate with NEXT and PREVIOUS buttons", 
                font=("Arial", 7), bg=self.COLORS['bg_light'], fg='#999999').pack(anchor=tk.W)
        
        # HOW TO PLAY section
        tk.Frame(info_frame, bg=self.COLORS['bg_light'], height=1).pack(fill=tk.X, pady=5)
        
        play_title = tk.Label(info_frame, text="🎮 HOW TO PLAY",
                             font=("Arial", 10, "bold"),
                             bg=self.COLORS['bg_light'],
                             fg=self.COLORS['accent1'])
        play_title.pack(pady=(5, 5))
        
        instructions = [
            "1. Click GENERATE to create board",
            "2. Click SOLVE to find solution (auto)",
            "3. Use NEXT/PREV to view steps",
            "4. Click PLAY to move manually",
            "5. Click adjacent cells to move"
        ]
        
        for instruction in instructions:
            instr_label = tk.Label(info_frame, text=instruction,
                                  font=("Arial", 7),
                                  bg=self.COLORS['bg_light'],
                                  fg='#bbbbbb',
                                  justify=tk.LEFT)
            instr_label.pack(anchor=tk.W, padx=12, pady=1)
        
        # KEYBOARD CONTROLS section
        tk.Frame(info_frame, bg=self.COLORS['bg_light'], height=1).pack(fill=tk.X, pady=5)
        
        kb_title = tk.Label(info_frame, text="⌨️ KEYBOARD CONTROLS",
                           font=("Arial", 10, "bold"),
                           bg=self.COLORS['bg_light'],
                           fg=self.COLORS['accent2'])
        kb_title.pack(pady=(5, 5))
        
        # UP movement
        up_frame = tk.Frame(info_frame, bg=self.COLORS['bg_light'])
        up_frame.pack(fill=tk.X, padx=12, pady=2)
        tk.Label(up_frame, text="↑ UP", font=("Arial", 7, "bold"),
                bg=self.COLORS['bg_light'], fg=self.COLORS['primary']).pack(side=tk.LEFT, padx=3)
        tk.Label(up_frame, text="W key | Cost +2", font=("Arial", 7),
                bg=self.COLORS['bg_light'], fg='#bbbbbb').pack(side=tk.LEFT)
        
        # DOWN movement
        down_frame = tk.Frame(info_frame, bg=self.COLORS['bg_light'])
        down_frame.pack(fill=tk.X, padx=12, pady=2)
        tk.Label(down_frame, text="↓ DOWN", font=("Arial", 7, "bold"),
                bg=self.COLORS['bg_light'], fg=self.COLORS['success']).pack(side=tk.LEFT, padx=3)
        tk.Label(down_frame, text="S key | Cost +0", font=("Arial", 7),
                bg=self.COLORS['bg_light'], fg='#bbbbbb').pack(side=tk.LEFT)
        
        # LEFT movement
        left_frame = tk.Frame(info_frame, bg=self.COLORS['bg_light'])
        left_frame.pack(fill=tk.X, padx=12, pady=2)
        tk.Label(left_frame, text="← LEFT", font=("Arial", 7, "bold"),
                bg=self.COLORS['bg_light'], fg=self.COLORS['warning']).pack(side=tk.LEFT, padx=3)
        tk.Label(left_frame, text="A key | Cost +1", font=("Arial", 7),
                bg=self.COLORS['bg_light'], fg='#bbbbbb').pack(side=tk.LEFT)
        
        # RIGHT movement
        right_frame = tk.Frame(info_frame, bg=self.COLORS['bg_light'])
        right_frame.pack(fill=tk.X, padx=12, pady=2)
        tk.Label(right_frame, text="→ RIGHT", font=("Arial", 7, "bold"),
                bg=self.COLORS['bg_light'], fg=self.COLORS['danger']).pack(side=tk.LEFT, padx=3)
        tk.Label(right_frame, text="D key | Cost +1", font=("Arial", 7),
                bg=self.COLORS['bg_light'], fg='#bbbbbb').pack(side=tk.LEFT)
        
        # Arrow keys alternative
        arrow_frame = tk.Frame(info_frame, bg=self.COLORS['bg_light'])
        arrow_frame.pack(fill=tk.X, padx=12, pady=4)
        tk.Label(arrow_frame, text="Or use Arrow Keys (↑↓←→) for movement",
                font=("Arial", 6, "italic"),
                bg=self.COLORS['bg_light'], fg='#888888').pack(anchor=tk.W)
    
    def create_stat_box(self, parent, label, value, color):
        """Create a stat box and return the value label"""
        frame = tk.Frame(parent, bg=self.COLORS['bg_dark'])
        
        label_font = font.Font(family="Arial", size=9)
        value_font = font.Font(family="Arial", size=18, weight="bold")
        
        tk.Label(frame, text=label, font=label_font,
                bg=self.COLORS['bg_dark'], fg=self.COLORS['white']).pack()
        
        stat_label = tk.Label(frame, text=value, font=value_font,
                             bg=self.COLORS['bg_dark'], fg=color)
        stat_label.pack()
        
        return stat_label
    
    def setup_board_section(self, parent):
        """Setup board visualization section"""
        board_frame = tk.Frame(parent, bg=self.COLORS['bg_dark'])
        board_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create rounded board container
        container = tk.Frame(board_frame, bg=self.COLORS['bg_light'], 
                            relief=tk.RAISED, bd=3)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.canvas = tk.Canvas(container, bg=self.COLORS['white'],
                               width=420, height=420,
                               highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        
        # Update canvas on window resize
        self.root.bind("<Configure>", self._on_window_resize)
    
    def setup_step_section(self, parent):
        """Setup step information section (deprecated - moved to right panel)"""
        pass
    
    def setup_buttons_section(self, parent):
        """Setup action buttons section (deprecated - moved to right panel)"""
        pass
    
    def create_modern_button(self, parent, text, command, color):
        """Create a modern styled button"""
        btn = tk.Button(parent, text=text, command=command,
                       bg=color, fg='#000000',
                       font=("Arial", 12, "bold"),
                       padx=20, pady=12,
                       relief=tk.FLAT, cursor='hand2',
                       activebackground=color,
                       activeforeground='#000000',
                       highlightthickness=0,
                       state=tk.NORMAL)
        return btn
    
    def on_canvas_click(self, event):
        """Handle click on canvas - manual movement in play mode"""
        if not self.board or self.solving_in_progress:
            return
        
        # If not in manual mode, start manual mode
        if not self.manual_play_enabled:
            self.manual_play_enabled = True
            self.manual_vacuum_pos = self.board.vacuum_pos
            self.manual_cost = 0
            self.manual_solution_path = []
            self.current_step = 0
            self.manual_mode_label.config(text="MANUAL MODE - Click cells to move", fg=self.COLORS['warning'])
            self.status_stat_label.config(text="PLAYING", fg=self.COLORS['warning'])
            self.detail_label.config(text="Click adjacent cells (up/down/left/right) to move the vacuum")
            self.draw_board(manual_mode=True)
            return
        
        # Use the same calculation method as draw_board for consistency
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # Use smaller dimension for square cells (same as draw_board)
        board_size = min(canvas_width, canvas_height) - 20
        if board_size < 100:
            board_size = 400  # Fallback
        
        cell_size = board_size // 6
        
        # Calculate which cell was clicked
        col = event.x // cell_size
        row = event.y // cell_size
        
        # Validate position on board
        if not (0 <= row < 6 and 0 <= col < 6):
            self.detail_label.config(text="Click within the board grid!", fg=self.COLORS['danger'])
            return
        
        clicked_pos = (row, col)
        current_row, current_col = self.manual_vacuum_pos
        
        # Check if it's the current position
        if clicked_pos == (current_row, current_col):
            self.detail_label.config(text="Already at this position!", fg=self.COLORS['warning'])
            return
        
        # Check if clicked cell is adjacent (only 1 step away)
        row_diff = abs(row - current_row)
        col_diff = abs(col - current_col)
        
        if row_diff + col_diff != 1:
            # Not adjacent - show distance
            distance = row_diff + col_diff
            self.detail_label.config(text=f"❌ Too far! ({distance} cells away) Click adjacent cell only!", fg=self.COLORS['danger'])
            return
        
        # Check if destination has obstacle
        if not self.board.is_valid_position(row, col):
            self.detail_label.config(text="❌ Cannot move to obstacle!", fg=self.COLORS['danger'])
            return
        
        # Calculate move direction and cost
        if row < current_row:
            direction = "UP"
            cost = 2
        elif row > current_row:
            direction = "DOWN"
            cost = 0
        elif col < current_col:
            direction = "LEFT"
            cost = 1
        else:  # col > current_col
            direction = "RIGHT"
            cost = 1
        
        # Update manual position and cost
        self.manual_vacuum_pos = clicked_pos
        self.manual_cost += cost
        
        # Add to manual solution path (like NEXT button does)
        move_entry = {
            'from': (current_row, current_col),
            'to': clicked_pos,
            'move': direction,
            'cost': cost
        }
        self.manual_solution_path.append(move_entry)
        
        # Increment step like NEXT button does
        self.current_step += 1
        
        # Update stats in real-time
        self.cost_stat_label.config(text=str(self.manual_cost))
        self.step_label.config(text=f"Step: {self.current_step}/{self.current_step}")
        
        # Check if reached goal
        if clicked_pos == self.board.dirt_pos:
            self.detail_label.config(text=f"🎉 GOAL REACHED! Total Cost: {self.manual_cost}", fg=self.COLORS['success'])
            self.manual_mode_label.config(text="✓ MANUAL MODE - GOAL REACHED!", fg=self.COLORS['success'])
            self.status_stat_label.config(text="WON!", fg=self.COLORS['success'])
            # Use manual_solution_path for highlighting
            self.solution_path = self.manual_solution_path
            messagebox.showinfo("Success!", f"You reached the goal!\n\nTotal Cost: {self.manual_cost}\n\nClick RESET or PLAY to try again")
        else:
            self.detail_label.config(text=f"✓ Moved {direction} (Cost +{cost}) | Total: {self.manual_cost}", fg=self.COLORS['success'])
            # Update next/prev buttons based on position
            self.btn_next.config(state=tk.DISABLED)
            self.btn_previous.config(state=tk.NORMAL)
        
        # Use manual_solution_path for highlighting - same as NEXT button
        self.solution_path = self.manual_solution_path
        self.draw_board(highlight_step=self.current_step)
    
    def on_key_press(self, event):
        """Handle keyboard input for manual play mode"""
        if not self.board or not self.manual_play_enabled or self.solving_in_progress:
            return
        
        current_row, current_col = self.manual_vacuum_pos
        
        # Determine direction based on key pressed
        # Try both keysym and char for better compatibility
        key = event.keysym.lower() if event.keysym else ""
        char = event.char.lower() if event.char else ""
        
        target_row, target_col = None, None
        direction = ""
        cost = 0
        
        # Check for UP movement
        if key in ['up', 'w'] or char in ['w']:
            target_row, target_col = current_row - 1, current_col
            direction = "UP"
            cost = 2
        # Check for DOWN movement
        elif key in ['down', 's'] or char in ['s']:
            target_row, target_col = current_row + 1, current_col
            direction = "DOWN"
            cost = 0
        # Check for LEFT movement
        elif key in ['left', 'a'] or char in ['a']:
            target_row, target_col = current_row, current_col - 1
            direction = "LEFT"
            cost = 1
        # Check for RIGHT movement
        elif key in ['right', 'd'] or char in ['d']:
            target_row, target_col = current_row, current_col + 1
            direction = "RIGHT"
            cost = 1
        
        # If no valid direction found, return
        if target_row is None or target_col is None:
            return
        
        # Check bounds
        if not (0 <= target_row < 6 and 0 <= target_col < 6):
            self.detail_label.config(text=f"❌ Cannot move {direction} - outside board!", fg=self.COLORS['danger'])
            return
        
        # Check if destination has obstacle
        if not self.board.is_valid_position(target_row, target_col):
            self.detail_label.config(text=f"❌ Cannot move {direction} - obstacle blocks path!", fg=self.COLORS['danger'])
            return
        
        # Update manual position and cost
        self.manual_vacuum_pos = (target_row, target_col)
        self.manual_cost += cost
        
        # Check if reached goal
        if (target_row, target_col) == self.board.dirt_pos:
            self.detail_label.config(text=f"🎉 GOAL REACHED! Total Cost: {self.manual_cost}", fg=self.COLORS['success'])
            self.manual_mode_label.config(text="MANUAL MODE - GOAL REACHED!", fg=self.COLORS['success'])
            messagebox.showinfo("Success!", f"You reached the goal!\n\nTotal Cost: {self.manual_cost}")
        else:
            self.detail_label.config(text=f"✓ Moved {direction} (Cost +{cost}) | Total: {self.manual_cost}", fg=self.COLORS['success'])
        
        self.draw_board(manual_mode=True)
    
    def _on_window_resize(self, event):
        """Redraw board when window is resized"""
        if self.board:
            self.draw_board(highlight_step=self.current_step if self.current_step > 0 else None)
    
    def generate_board(self):
        """Generate a new random board"""
        self.board = Board(width=6, height=6)
        self.board.generate_random(obstacle_count=8)
        
        self.solution_path = []
        self.current_step = 0
        self.total_cost = 0
        self.solver = None
        self.manual_solution_path = []
        self.manual_play_enabled = False
        
        self.btn_solve.config(state=tk.NORMAL)
        self.btn_previous.config(state=tk.DISABLED)
        self.btn_next.config(state=tk.DISABLED)
        self.btn_reset.config(state=tk.DISABLED)
        
        self.status_stat_label.config(text="READY", fg=self.COLORS['success'])
        self.moves_stat_label.config(text="0")
        self.cost_stat_label.config(text="0")
        self.step_label.config(text="0/0")
        self.detail_label.config(text="Click SOLVE to find the path")
        
        self.draw_board()
    
    def shuffle_board(self):
        """Shuffle board (similar to generate but different)"""
        self.generate_board()
        self.detail_label.config(text="Board shuffled!")
    
    def start_manual_play(self):
        """Start manual play mode"""
        if not self.board:
            messagebox.showwarning("Warning", "Generate a board first!")
            return
        
        # Initialize manual play variables
        self.manual_play_enabled = True
        self.manual_vacuum_pos = self.board.vacuum_pos
        self.manual_cost = 0
        self.manual_solution_path = []
        
        # Update UI with CLICK instructions
        self.manual_mode_label.config(text="MANUAL MODE - CLICK CELLS TO MOVE!", fg=self.COLORS['warning'])
        self.status_stat_label.config(text="PLAYING", fg=self.COLORS['warning'])
        self.detail_label.config(text="👆 Click an adjacent cell (up/down/left/right) to move", fg=self.COLORS['accent2'])
        self.step_label.config(text="Click adjacent cells to move vacuum")
        
        # Clear solution display
        self.solution_path = []
        self.current_step = 0
        self.cost_stat_label.config(text="0")
        self.moves_stat_label.config(text="0")
        self.btn_previous.config(state=tk.DISABLED)
        self.btn_next.config(state=tk.DISABLED)
        
        # Disable solve button during manual play
        self.btn_solve.config(state=tk.DISABLED)
        self.btn_reset.config(state=tk.NORMAL)
        
        # Set focus to canvas for clicks
        self.canvas.focus_set()
        
        self.draw_board()
    
    def solve(self):
        """Solve using DFS in background thread"""
        if not self.board:
            messagebox.showwarning("Warning", "Generate a board first!")
            return
        
        if self.solving_in_progress:
            return
        
        self.solving_in_progress = True
        self.btn_solve.config(state=tk.DISABLED)
        self.btn_generate.config(state=tk.DISABLED)
        self.btn_shuffle.config(state=tk.DISABLED)
        self.btn_previous.config(state=tk.DISABLED)
        self.btn_next.config(state=tk.DISABLED)
        self.status_stat_label.config(text="SOLVING", fg=self.COLORS['warning'])
        self.detail_label.config(text="Finding solution with DFS...")
        
        # Run solver in background
        thread = threading.Thread(target=self._solve_background)
        thread.daemon = True
        thread.start()
    
    def _solve_background(self):
        """Background thread for solving"""
        self.solver = VacuumDFS(self.board)
        success, path, cost, message = self.solver.solve()
        
        if success:
            self.solution_path = self.solver.path
            self.total_cost = cost
            self.root.after(0, self._update_after_solve, True, len(path))
        else:
            self.root.after(0, self._update_after_solve, False, 0)
    
    def _update_after_solve(self, success, move_count):
        """Update UI after solving"""
        self.solving_in_progress = False
        self.btn_generate.config(state=tk.NORMAL)
        self.btn_solve.config(state=tk.NORMAL)
        self.btn_shuffle.config(state=tk.NORMAL)
        self.btn_reset.config(state=tk.NORMAL)
        
        if success:
            self.status_stat_label.config(text="SOLVED!", fg=self.COLORS['success'])
            self.cost_stat_label.config(text=str(self.total_cost))
            self.moves_stat_label.config(text=str(move_count))
            self.step_label.config(text=f"0/{move_count}")
            self.detail_label.config(text="Solution found! Click NEXT to view steps")
            self.btn_next.config(state=tk.NORMAL)
            
            # Write solution to file with correct parameters
            writer = SolutionWriter()
            writer.write(self.board, self.solution_path, self.total_cost, True, self.solver)
        else:
            self.status_stat_label.config(text="NO SOLUTION", fg=self.COLORS['danger'])
            self.detail_label.config(text="No path found - obstacles block the way")
            
            # Write "no solution" message to file
            writer = SolutionWriter()
            writer.write(self.board, [], 0, False, self.solver)
        
        self.draw_board()
    
    def next_step(self):
        """Move to next step in solution"""
        if not self.solution_path or self.current_step >= len(self.solution_path):
            return
        
        self.current_step += 1
        self.update_step_display()
        
        # Update buttons
        if self.current_step >= len(self.solution_path):
            self.btn_next.config(state=tk.DISABLED)
        self.btn_previous.config(state=tk.NORMAL)
    
    def previous_step(self):
        """Move to previous step in solution"""
        if self.current_step <= 0:
            return
        
        self.current_step -= 1
        self.update_step_display()
        
        # Update buttons
        if self.current_step <= 0:
            self.btn_previous.config(state=tk.DISABLED)
        self.btn_next.config(state=tk.NORMAL)
    
    def reset_solution(self):
        """Reset to start of solution"""
        # Handle manual play mode
        if self.manual_play_enabled:
            self.manual_vacuum_pos = self.board.vacuum_pos
            self.manual_cost = 0
            self.manual_solution_path = []
            self.current_step = 0
            self.cost_stat_label.config(text="0")
            self.detail_label.config(text="👆 Click an adjacent cell to move")
            self.btn_previous.config(state=tk.DISABLED)
            self.btn_next.config(state=tk.DISABLED)
            self.draw_board()
            return
        
        # Handle auto solve mode
        if self.solution_path:
            self.current_step = 0
            self.btn_previous.config(state=tk.DISABLED)
            self.btn_next.config(state=tk.NORMAL)
            self.update_step_display()
    
    def update_step_display(self):
        """Update display for current step"""
        total = len(self.solution_path)
        self.step_label.config(text=f"Step: {self.current_step}/{total}")
        
        if self.current_step == 0:
            self.detail_label.config(text="Starting position")
        elif self.current_step > 0 and self.current_step <= len(self.solution_path):
            move = self.solution_path[self.current_step - 1]
            direction = move['move']
            cost = move['cost']
            self.detail_label.config(text=f"Move {direction} (Cost: {cost})")
        
        self.draw_board(highlight_step=self.current_step)
    
    def draw_board(self, highlight_step=None, manual_mode=False):
        """Draw the board on canvas"""
        self.canvas.delete("all")
        
        if not self.board:
            return
        
        # Get actual canvas size
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # Use smaller dimension for square cells
        board_size = min(canvas_width, canvas_height) - 20
        if board_size < 100:
            board_size = 400  # Fallback
        
        cell_size = board_size // 6
        
        # Draw grid
        for row in range(6):
            for col in range(6):
                x1 = col * cell_size
                y1 = row * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size
                
                # Determine cell color
                cell_value = self.board.grid[row][col]
                
                # In manual mode, show manual vacuum position instead of board vacuum
                if manual_mode and self.manual_vacuum_pos == (row, col):
                    color = self.COLORS['vacuum']
                elif highlight_step and highlight_step > 0 and highlight_step <= len(self.solution_path):
                    # Check if this is on the path
                    move = self.solution_path[highlight_step - 1]
                    target_row, target_col = move['to']
                    
                    if row == target_row and col == target_col:
                        color = self.COLORS['path']
                    elif cell_value == Board.OBSTACLE:
                        color = self.COLORS['obstacle']
                    elif cell_value == Board.DIRT:
                        color = self.COLORS['dirt']
                    elif cell_value == Board.VACUUM:
                        color = self.COLORS['vacuum']
                    else:
                        color = self.COLORS['empty']
                else:
                    if cell_value == Board.OBSTACLE:
                        color = self.COLORS['obstacle']
                    elif cell_value == Board.DIRT:
                        color = self.COLORS['dirt']
                    elif cell_value == Board.VACUUM and not manual_mode:
                        color = self.COLORS['vacuum']
                    else:
                        color = self.COLORS['empty']
                
                # Draw cell
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color,
                                            outline='#cccccc', width=1)
                
                # Draw text
                if manual_mode and self.manual_vacuum_pos == (row, col):
                    text = "V"
                elif cell_value == Board.OBSTACLE:
                    text = "#"
                elif cell_value == Board.DIRT:
                    text = "G"
                elif cell_value == Board.VACUUM and not manual_mode:
                    text = "V"
                else:
                    text = ""
                
                if text:
                    self.canvas.create_text(x1 + cell_size//2, y1 + cell_size//2,
                                           text=text, font=("Arial", 16, "bold"),
                                           fill='white')


def main():
    root = tk.Tk()
    gui = VacuumGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
