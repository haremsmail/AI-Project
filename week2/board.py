import random


class Board:
    """Represents the grid-based environment"""
    
    EMPTY = 0
    OBSTACLE = 1
    DIRT = 2
    VACUUM = 3
    
    SYMBOLS = {
        EMPTY: ".",
        OBSTACLE: "#",
        DIRT: "G",
        VACUUM: "V"
    }
    
    def __init__(self, width=6, height=6):
        self.width = width
        self.height = height
        self.grid = [[Board.EMPTY for _ in range(width)] for _ in range(height)]
        self.vacuum_pos = None
        self.dirt_pos = None
        
    def generate_random(self, obstacle_count=6):
        """Generate random board with vacuum, dirt, and obstacles"""
        # Reset grid
        self.grid = [[Board.EMPTY for _ in range(self.width)] for _ in range(self.height)]
        
        # Place vacuum at random position
        self.vacuum_pos = (random.randint(0, self.height-1), random.randint(0, self.width-1))
        self.grid[self.vacuum_pos[0]][self.vacuum_pos[1]] = Board.VACUUM
        
        # Place dirt at different position
        while True:
            dirt_pos = (random.randint(0, self.height-1), random.randint(0, self.width-1))
            if dirt_pos != self.vacuum_pos:
                self.dirt_pos = dirt_pos
                self.grid[self.dirt_pos[0]][self.dirt_pos[1]] = Board.DIRT
                break
        
        # Place obstacles
        obstacles_placed = 0
        while obstacles_placed < obstacle_count:
            obs_pos = (random.randint(0, self.height-1), random.randint(0, self.width-1))
            if obs_pos != self.vacuum_pos and obs_pos != self.dirt_pos:
                self.grid[obs_pos[0]][obs_pos[1]] = Board.OBSTACLE
                obstacles_placed += 1
    
    def is_valid_position(self, row, col):
        """Check if position is within bounds and not an obstacle"""
        return (0 <= row < self.height and 
                0 <= col < self.width and 
                self.grid[row][col] != Board.OBSTACLE)
    
    def get_symbol(self, row, col):
        """Get symbol for display"""
        return Board.SYMBOLS.get(self.grid[row][col], "?")
    
    def to_string(self):
        """Convert board to string representation"""
        lines = []
        for row in self.grid:
            line = " ".join(Board.SYMBOLS[cell] for cell in row)
            lines.append(line)
        return "\n".join(lines)
    
    def copy(self):
        """Create a copy of the board"""
        new_board = Board(self.width, self.height)
        new_board.grid = [row[:] for row in self.grid]
        new_board.vacuum_pos = self.vacuum_pos
        new_board.dirt_pos = self.dirt_pos
        return new_board
