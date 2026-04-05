import random
""" this file create 6*6 bord of puzzle """


class Board:
    """Represents the grid-based environment"""
    
    """ har xanayak la torakay da zhmarakay haya"""
    EMPTY = 0
    OBSTACLE = 1
    DIRT = 2
    VACUUM = 3
    """" used to for manage each number in gird"""
    
    SYMBOLS = {
        EMPTY: ".",
        OBSTACLE: "#",
        DIRT: "G",
        VACUUM: "V"
    }
    """ convert the number to the symbol  numana agar sfr bu daygor ba batal ."""
    
    def __init__(self, width=6, height=6):
        """ this is have conscutor have bord size   by deffaul 6 by 6"""
        self.width = width
        self.height = height
        self.grid = [[Board.EMPTY for _ in range(width)] for _ in range(height)]
        """ auayn two dimantion gridak drust dakaa it mean  bo nmuna agar bley widht 6 height 6 
        [
 [0,0,),0,0,0],
 [0,0,0,0,0,0],
 [0,0,0,0,0,0],
 [0,0,0,0,0,0],
 [0,0,0,0,0,0],
 [0,0,0,0,0,0]
]"""

        self.vacuum_pos = None
        self.dirt_pos = None
        """" bakar de bo zyaytkdny postion hardukayn vacuum w dirt  """
    def generate_random(self, obstacle_count=6):
        """" bordeky haramaky drust daak"""
        """Generate random board with vacuum, dirt, and obstacles"""
        # Reset grid kkkkk
        self.grid = [[Board.EMPTY for _ in range(self.width)] for _ in range(self.height)]
        
        
        # Place vacuum at random position
        self.vacuum_pos = (random.randint(0, self.height-1), random.randint(0, self.width-1))
        """ labar auay arrray start zero bka 0-5 6-1=5 0-5 6-1=5"""
        self.grid[self.vacuum_pos[0]][self.vacuum_pos[1]] = Board.VACUUM
        """ mabasty la 0 1 row colum  nauak index sfru index 1"""
        """. . . . . .
. . . . . .
. . . . V .
. . . . . .
. . . . . .
. . . . . ."""

        # Place dirt at different position
        while True:
            dirt_pos = (random.randint(0, self.height-1), random.randint(0, self.width-1))
            if dirt_pos != self.vacuum_pos:
                """" dlnayay dada dirtu pos yaksan nya"""
                self.dirt_pos = dirt_pos
                self.grid[self.dirt_pos[0]][self.dirt_pos[1]] = Board.DIRT
                break
            """ wata shueny dirt dabynauta stop daak"""
        
        # Place obstacles astnagkan
        obstacles_placed = 0
        while obstacles_placed < obstacle_count:
            """ barduam la danany barbastakan ta dagata zhmaray pe wyst"""
            obs_pos = (random.randint(0, self.height-1), random.randint(0, self.width-1))
            if obs_pos != self.vacuum_pos and obs_pos != self.dirt_pos:
                self.grid[obs_pos[0]][obs_pos[1]] = Board.OBSTACLE
                """ image obtracl dadane"""
                obstacles_placed += 1
    
    def is_valid_position(self, row, col):
        """ wata vacum datuane brua bo aua postiona yaxu na"""
        """Check if position is within bounds and not an obstacle"""
        return (0 <= row < self.height and 
                0 <= col < self.width and 
                self.grid[row][col] != Board.OBSTACLE)
    

    """can you vacume go to there or not"""
    """ wata agar la shapaka nahata daraua yaxu yaksan nabe ba   batbast aua esh daka"""
    """ nmuna agar royaka -1  aua outside """
  
  
  
  
    def get_symbol(self, row, col):
        """Get symbol for displa lau functiala zhamra dagore bo symbol"""
        return Board.SYMBOLS.get(self.grid[row][col], "?")
    """ agar value exit nabur return au nyshana psyara daaa"""
    
    def to_string(self):
        """Convert board to string representation"""
        lines = []
        """ wata bordaka zhmarakan dakata  daqeky juana"""
        for row in self.grid:
            line = " ".join(Board.SYMBOLS[cell] for cell in row)
            """ single value inside the row """
            lines.append(line)
        return "\n".join(lines)
    """
    [
 [0,0,0],
 [0,3,0],
 [0,2,0]
]
this change to
. . .
. V .
. G .
"""
    
    def copy(self):
        """ bordkey nwe drust daka hamn bordy esta"""
        """Create a copy of the board"""
        new_board = Board(self.width, self.height)
        new_board.grid = [row[:] for row in self.grid]
        """ it is a copy of each row"""
        new_board.vacuum_pos = self.vacuum_pos
        new_board.dirt_pos = self.dirt_pos
        return new_board
 