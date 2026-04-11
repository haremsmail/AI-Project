
import tkinter as tk
from tkinter import messagebox
""" bakar de bo pishandany alertek yan shtek"""
import math
from graph import Graph
""" implmenty grafh class kraua"""
from astar import AStarFinder
""""""

class AStarGUI:
    
    
    COLORS = {
        'bg': '#0a1628', 'light': '#1a2f5a', 'cyan': '#00d4ff',
        'green': '#00ff88', 'red': '#ff3366', 'orange': '#ffaa00',
        'blue': '#3366ff', 'gray': '#666666', 'grid': '#333333'
    }
    
    def __init__(self, root):
        self.root = root
        """root = tk.Tk()"""
        self.root.title("A* Pathfinding")
        self.root.geometry("1100x1000")
        self.root.configure(bg=self.COLORS['bg'])
        """auayan bo backgorund color"""
        
        self.graph = Graph()
        self.astar = AStarFinder()
        self.selected_node = None
        self.start_node = self.goal_node = None
        """ start node nody daspekrdn goal node au noday pe dagay"""
        self.solution_path = self.explored_nodes = []
        self.positions = {}
        """auayan story au nodana daka ka drow dakren"""
        self.radius = 18
        """ auayn size har nodeka"""
        self.grid_size = 50
        """ auayn space newuan nodakan"""
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup interface"""
        tk.Label(self.root, text="A* PATHFINDING", font=("Arial", 24, "bold"),
                bg=self.COLORS['bg'], fg=self.COLORS['cyan']).pack(pady=10)
        """ create big title daka la top"""
        
        main = tk.Frame(self.root, bg=self.COLORS['bg'])
        """ auayn bo drusktnry contianer yaxud waku boxeka"""
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Canvas (left)
        left = tk.Frame(main, bg=self.COLORS['bg'])
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        canvas_frame = tk.Frame(left, bg=self.COLORS['light'], relief=tk.RAISED, bd=2)
        """ auayn snureky barz krau dadat bd wata astury stun"""
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(canvas_frame, bg='#f5f5f5', width=580, height=500, highlightthickness=0)
        """ lerada ka hamu shtek keshra Apathing edge node au ana  hilightthinkes space outline"""
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Button-3>", lambda e: setattr(self, 'selected_node', None) or self.redraw())
        """" au du button lo select krdny node bakar de ca clcikt labutton 3 krd dubara rdrow cnavas daka
        """

        """ simple canvas what is it drow graph"""
        
        self.draw_grid()
        """ drow backgorund gird of canvase"""
        
        # Controls (right)
        right = tk.Frame(main, bg=self.COLORS['bg'], width=250)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        """ padx space from left side"""
        right.pack_propagate(False)
        """ regry daka la goryny qabaraka"""
        
        # Add node section
        self.section(right, "ADD NODE")
        tk.Label(right, text="Name:", bg=self.COLORS['bg'], fg='white', font=("Arial", 9)).pack(anchor=tk.W, padx=10)
        self.name_entry = tk.Entry(right, width=20)
        self.name_entry.pack(padx=10, pady=(0, 2))
        
        tk.Label(right, text="X:", bg=self.COLORS['bg'], fg='white', font=("Arial", 9)).pack(anchor=tk.W, padx=10)
        self.x_entry = tk.Entry(right, width=20)
        self.x_entry.pack(padx=10, pady=(0, 2))
        
        tk.Label(right, text="Y:", bg=self.COLORS['bg'], fg='white', font=("Arial", 9)).pack(anchor=tk.W, padx=10)
        self.y_entry = tk.Entry(right, width=20)
        self.y_entry.pack(padx=10, pady=(0, 3))
        
        self.button(right, "ADD NODE", self.add_node, self.COLORS['green'])
        """ agar click le krd bangy funciton add-nde hq4 lq ril3 tui"""

        

        # Graph stats auayn riight node edgt
        self.section(right, "GRAPH")
        tk.Label(right, text="Nodes:", bg=self.COLORS['bg'], fg='white', font=("Arial", 9)).pack(anchor=tk.W, padx=10)
        self.nodes_label = tk.Label(right, text="0", font=("Arial", 14, "bold"), bg=self.COLORS['bg'], fg=self.COLORS['cyan'])
        self.nodes_label.pack(pady=(0, 4))
        
        tk.Label(right, text="Edges:", bg=self.COLORS['bg'], fg='white', font=("Arial", 9)).pack(anchor=tk.W, padx=10)
        self.edges_label = tk.Label(right, text="0", font=("Arial", 14, "bold"), bg=self.COLORS['bg'], fg=self.COLORS['orange'])
        self.edges_label.pack(pady=(0, 4))
        """ duatr la regay function auana zyadu kam dakan"""
        
        # A* search
        self.section(right, "A* SEARCH")
        """ postion right aserach pishan dada la bottom"""
        tk.Label(right, text="Start:", bg=self.COLORS['bg'], fg='white', font=("Arial", 9)).pack(anchor=tk.W, padx=10)
        self.start_var = tk.StringVar(value="S")
        self.start_menu = tk.OptionMenu(right, self.start_var, "S")
        self.start_menu.config(bg=self.COLORS['blue'], fg='white', font=("Arial", 8), width=20)
        self.start_menu.pack(fill=tk.X, padx=10, pady=(0, 2))
        
        tk.Label(right, text="Goal:", bg=self.COLORS['bg'], fg='white', font=("Arial", 9)).pack(anchor=tk.W, padx=10)
        self.goal_var = tk.StringVar(value="G")
        self.goal_menu = tk.OptionMenu(right, self.goal_var, "G")
        self.goal_menu.config(bg=self.COLORS['red'], fg='white', font=("Arial", 8), width=20)
        self.goal_menu.pack(fill=tk.X, padx=10, pady=(0, 3))
        """ place on ui tk.x basheuy asoy keshanh"""
        
        self.button(right, "FIND PATH", self.solve, self.COLORS['cyan'])
        
        # Results
        self.section(right, "RESULTS")
        tk.Label(right, text="Path:", bg=self.COLORS['bg'], fg='white', font=("Arial", 9)).pack(anchor=tk.W, padx=10)
        self.path_label = tk.Label(right, text="—", font=("Arial", 8), bg=self.COLORS['bg'], 
                                   fg=self.COLORS['orange'], wraplength=220)
        self.path_label.pack(pady=(0, 1), padx=10)
        
        tk.Label(right, text="Cost:", bg=self.COLORS['bg'], fg='white', font=("Arial", 9)).pack(anchor=tk.W, padx=10)
        self.cost_label = tk.Label(right, text="0", font=("Arial", 12, "bold"), bg=self.COLORS['bg'], fg=self.COLORS['green'])
        self.cost_label.pack(pady=(0, 1))
        
        tk.Label(right, text="Explored:", bg=self.COLORS['bg'], fg='white', font=("Arial", 9)).pack(anchor=tk.W, padx=10)
        """ wata zhmary nodakan pesh aauya to bashtryn path halbzhery"""
        self.explored_label = tk.Label(right, text="0", font=("Arial", 12, "bold"), bg=self.COLORS['bg'], fg=self.COLORS['orange'])
        self.explored_label.pack(pady=(0, 3))
        
        # Action buttons
        btn_frame = tk.Frame(right, bg=self.COLORS['bg'])
        btn_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Button(btn_frame, text="SAMPLE", font=("Arial", 8, "bold"), bg=self.COLORS['cyan'], fg='black',
                 command=self.load_sample, relief=tk.RAISED).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))
        tk.Button(btn_frame, text="DETAILS", font=("Arial", 8, "bold"), bg=self.COLORS['orange'], fg='black',
                 command=self.show_details, relief=tk.RAISED).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3)
        tk.Button(btn_frame, text="RESET", font=("Arial", 8, "bold"), bg=self.COLORS['red'], fg='white',
                 command=self.reset, relief=tk.RAISED).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(3, 0))
    


    """ bo zyad krdny seciton nweya"""
    def section(self, parent, title):
        """ two parameter parent location title shtakas"""
        """Section separator"""
        tk.Frame(parent, bg=self.COLORS['light'], height=1).pack(fill=tk.X, padx=10, pady=3)
        tk.Label(parent, text=title, font=("Arial", 10, "bold"), bg=self.COLORS['bg'], 
                fg=self.COLORS['cyan']).pack(anchor=tk.W, padx=10, pady=(1, 3))
    
    def button(self, parent, text, cmd, color):
        """ auay yakam location auay tr texty button auay tr agar click krd chbka"""
        """Create button"""
        tk.Button(parent, text=text, font=("Arial", 9, "bold"), bg=color, 
                 fg='black' if color != self.COLORS['red'] else 'white',
                 
                 command=cmd, relief=tk.RAISED, bd=1).pack(fill=tk.X, padx=10, pady=5)
    """ agar  button red nabu text black agar red bu text white"""
    """ auayan mabasty auay 50 50 brtua"""
    def draw_grid(self):
        """Draw grid"""
        """ drow line verticka horizton """
        for x in range(0, 600, self.grid_size):
          
            self.canvas.create_line(x, 0, x, 600, fill=self.COLORS['grid'])
        for y in range(0, 600, self.grid_size):
            self.canvas.create_line(0, y, 600, y, fill=self.COLORS['grid'])
                  


    def add_node(self):
        """Add node"""
        try:
            name = self.name_entry.get().strip()
            """ data la input warbgra space la bda"""
            x, y = float(self.x_entry.get()), float(self.y_entry.get())
            
            if not name or x < 0 or x > 10 or y < 0 or y > 10:
                """ nabe be name be dabe la 0 lo 10 bo xu ya"""
                messagebox.showerror("Error", "Invalid input")
                return
            
            if self.graph.get_node(name):
                """ agar node aleady exits bu erro dada"""
                messagebox.showerror("Error", f"Node '{name}' exists")
                return
            
            self.graph.add_node(name, x, y)
            self.positions[name] = (x * self.grid_size, y * self.grid_size)
            self.name_entry.delete(0, tk.END)
            self.x_entry.delete(0, tk.END)
            self.y_entry.delete(0, tk.END)
            """ clear  input daka bo auay nodeky tr daxl bkay"""
            self.redraw()
            self.update_dropdowns()
        except ValueError:
            messagebox.showerror("Error", "Enter valid numbers")
    

    """ lerar ra auastm"""
    def redraw(self):
        """" auayan ba kard e bo refresh kranayy hamu sht lanau canvas"""
        """Redraw canvas"""
        self.canvas.delete("all")
        self.draw_grid()


        
        # Draw edges
        for node in self.graph.get_all_nodes():
            """ drustkrndy line ba hamu greaky nodakan teparent"""
            for neighbor in node.neighbors:
                """ loop laragey jiranakanaua"""

                x1, y1 = self.positions.get(node.name, (0, 0))
                """ auayn current nod auay ley ham nauy node lagak postion agar nabu default 0,0 wardagre"""
                x2, y2 = self.positions.get(neighbor.name, (0, 0))
                """ auayan nody duam ka konty dakay"""
                is_path = (len(self.solution_path) > 1 and node.name in self.solution_path and 
                          neighbor.name in self.solution_path and 
                          abs(self.solution_path.index(node.name) - self.solution_path.index(neighbor.name)) == 1)
                color = self.COLORS['green'] if is_path else self.COLORS['gray']
                """ dabe soluction path garuart be laya harduk nodaka lanau au patha bunyan habe  diffrenakash
                mabasty auaya ka dabe nodey duay xoy be """
                width = 3 if is_path else 1
                """ auashayn bo thinkess depend on path"""
                self.canvas.create_line(x1, y1, x2, y2, fill=color, width=width)
        
        # Draw nodes
        for node in self.graph.get_all_nodes():
            """loop daka basr hamu node"""
            x, y = self.positions.get(node.name, (0, 0))
            """ x,y dadpz"""
            
            if node.name == self.start_node:
                color = self.COLORS['cyan']
            elif node.name == self.goal_node:
                color = self.COLORS['red']
            elif node.name in self.solution_path:
                color = self.COLORS['green']
            elif node.name in self.explored_nodes:
                """ au nodanay la katy garan lekolynauy lasar krabe"""
                color = self.COLORS['orange']
            else:
                color = self.COLORS['blue']
            
            self.canvas.create_oval(x - self.radius, y - self.radius, 
                                   x + self.radius, y + self.radius, fill=color, outline='white', width=2)
            """ drow a circle node"""
            self.canvas.create_text(x, y, text=node.name, font=("Arial", 11, "bold"), fill='white')
            """agadary aua texty nau nodakay"""
        
        self.nodes_label.config(text=str(self.graph.get_node_count()))
        self.edges_label.config(text=str(self.graph.get_edge_count()))
        """ updated lable node edigth auayan zhmary node lagal edge"""
    
    def on_click(self, event):
        """ bo drust krndy line lanauen harduk node"""
        """ auayan drop dawn menu  katek click le dakay"""
        """Handle click"""
        clicked = self.get_node_at(event.x, event.y)
        """ x,y lanau event """
        if not clicked:
            return
        
        if not self.selected_node:
            self.selected_node = clicked
        elif self.selected_node == clicked:
            self.selected_node = None
            """auayn user click dka la diffrent node agar haman click lebka disclick"""
        else:
            self.graph.add_edge(self.selected_node, clicked, bidirectional=False)
            """ agar du node connect bua"""
            self.selected_node = None
            self.redraw()
            self.update_dropdowns()
            """ agaretau by default bary jarnay xoy"""
    


    def get_node_at(self, x, y):

        """Get node at position wata clickt lasar kam node krdua"""
        for name, (nx, ny) in self.positions.items():
            if math.sqrt((x - nx)**2 + (y - ny)**2) <= self.radius:
                return name
            """ auayan distance newaun nodakan calclualte daka agar lanau baznaka clickt kr au dagarentya"""
        return None
    """ wata agar clicl lanau bazna ka krdny return nod edaka agar na return hich"""
    
    def update_dropdowns(self):
        """Update menus"""
        names = [n.name for n in self.graph.get_all_nodes()]
        
        self.start_menu["menu"].delete(0, tk.END)
        """ auayn remove old item daka"""
        for name in names:
            self.start_menu["menu"].add_command(label=name, command=lambda n=name: self.start_var.set(n))
     
        self.goal_menu["menu"].delete(0, tk.END)
        """ agar clikc la satr kra option bgor"""
        """ clieaer goal """
        for name in names:
            self.goal_menu["menu"].add_command(label=name, command=lambda n=name: self.goal_var.set(n))
        
        if names:
            self.start_var.set(names[0])
            """ wata agr node habu fifst statt  axir goal"""
            self.goal_var.set(names[-1])
    



    def solve(self):
        """Run A*"""
        """ wata agar click la button find path aua esh"""
        start = self.graph.get_node(self.start_var.get())

        goal = self.graph.get_node(self.goal_var.get())

        
        if not start or not goal:
            messagebox.showerror("Error", "Select valid nodes")
            return
        
        self.graph.reset_all_costs()
        """ reet costu """
        self.start_node = start.name
        self.goal_node = goal.name
        self.solution_path, self.explored_nodes, cost = self.astar.find_path(start, goal)
        """ explored node au nodaya ka hamu pishan dad"""
        if self.solution_path:
            self.path_label.config(text=" → ".join(self.solution_path), fg=self.COLORS['green'])
            self.cost_label.config(text=f"{cost:.2f}")
            """ auayan boo pishan dnay costu paht"""
        else:
            self.path_label.config(text="No path!", fg=self.COLORS['red'])
            self.cost_label.config(text="∞")
        
        self.explored_label.config(text=str(len(self.explored_nodes)))
        self.redraw()
        """ zhmary au nodanay ka haya hamuy pishan dada"""
    


    def show_details(self):
        """Show details"""
        if not self.explored_nodes:
            messagebox.showinfo("Info", "Run A* first")
            """ masseg box psihan da"""
            return
        
        win = tk.Toplevel(self.root)
        win.title("A* Details")
        win.geometry("500x400")
        win.configure(bg=self.COLORS['bg'])
        
        tk.Label(win, text="EXPLORATION LOG", font=("Arial", 12, "bold"),
                bg=self.COLORS['bg'], fg=self.COLORS['cyan']).pack(pady=5)
        
        frame = tk.Frame(win, bg=self.COLORS['bg'])
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        scroll = tk.Scrollbar(frame)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        text = tk.Text(frame, bg=self.COLORS['light'], fg='white', font=("Courier", 8),
                      yscrollcommand=scroll.set, relief=tk.FLAT, bd=0)
        text.pack(fill=tk.BOTH, expand=True)
        scroll.config(command=text.yview)
        
        text.insert(tk.END, f"Start: {self.start_node} | Goal: {self.goal_node}\n")
        text.insert(tk.END, f"Path: {' → '.join(self.solution_path) if self.solution_path else 'None'}\n")
        text.insert(tk.END, f"Cost: {self.astar.total_cost:.2f} | Explored: {len(self.explored_nodes)}\n\n")
        text.insert(tk.END, "=" * 40 + "\n\n")
        
        for i, node_name in enumerate(self.explored_nodes, 1):
            node = self.graph.get_node(node_name)
            text.insert(tk.END, f"{i}. {node_name} at ({node.x}, {node.y})\n   g={node.g:.2f} h={node.h:.2f} f={node.f:.2f}\n\n")
        
        text.config(state=tk.DISABLED)
        """g → cost from Start to current node
h → estimated distance to Goal
f → total score
👉 f = g + h"""
    
    def reset(self):
        """Reset"""
        self.graph.clear()
        self.solution_path = self.explored_nodes = []
        self.positions.clear()
        self.start_node = self.goal_node = self.selected_node = None
        self.name_entry.delete(0, tk.END)
        self.x_entry.delete(0, tk.END)
        self.y_entry.delete(0, tk.END)
        self.redraw()
        self.update_dropdowns()
    
    def load_sample(self):
        """Load sample"""
        self.reset()
        
        nodes = [("S", 0, 0), ("A", 1, 2), ("B", 2, 1), ("C", 3, 3), ("G", 5, 5)]
        edges = [("S", "A"), ("S", "B"), ("A", "C"), ("B", "C"), ("C", "G")]
        
        for name, x, y in nodes:
            self.graph.add_node(name, x, y)
            self.positions[name] = (x * self.grid_size, y * self.grid_size)
        
        for n1, n2 in edges:
            self.graph.add_edge(n1, n2, bidirectional=False)
        
        self.start_var.set("S")
        self.goal_var.set("G")
        self.redraw()
        messagebox.showinfo("Sample Loaded", "S→A, S→B, A→C, B→C, C→G")


def main():
    """Main"""
    root = tk.Tk()
    gui = AStarGUI(root)
    root.mainloop()
