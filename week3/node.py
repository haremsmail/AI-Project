"""Node class for A* graph"""
import math


class Node:
    """Node in graph for A* pathfinding"""
    
    def __init__(self, name, x, y):
        """Initialize node"""
        self.name = name
        self.x = x
        self.y = y
        self.neighbors = []
        self.g = float('inf') 
        """ real distance from start to currnet node"""
        """ cost from start to t this node  deffualt inf"""
        self.h = 0
        """ h distance to agel"""
        self.f = float('inf')
        
        self.parent = None
        
    def add_neighbor(self, neighbor_node):
        """Add edge to neighbor"""
        if neighbor_node not in self.neighbors:
            self.neighbors.append(neighbor_node)
    
    def remove_neighbor(self, neighbor_node):
        """Remove edge to neighbor"""
        if neighbor_node in self.neighbors:
            self.neighbors.remove(neighbor_node)
    
    def euclidean_distance(self, other_node):
        """Calculate distance to another node"""
        return math.sqrt((other_node.x - self.x)**2 + (other_node.y - self.y)**2)
    
    def calculate_heuristic(self, goal_node):
        """Calculate heuristic h to goal"""
        self.h = self.euclidean_distance(goal_node)
    
    def update_f_cost(self):
        """Update f = g + h"""
        self.f = self.g + self.h
    
    def reset_costs(self):
        """Reset costs to initial values"""
        self.g = float('inf')
        self.h = 0
        self.f = float('inf')
        self.parent = None
        """ used to reset all cost in a algoriht"""
    
    def __lt__(self, other):
        """Compare by f cost"""
        return self.f < other.f
    """ smaller is better a valuey bchture bu """
    
    def __eq__(self, other):
        """Check if same node"""
        if isinstance(other, Node):
            return self.name == other.name
        return False
    """ auayn check daka ka aya du node yskann yan na"""
    
    def __hash__(self):
        """Hash for use in sets"""
        return hash(self.name)
    """ yahni lanauanay nauy ai bkata hashed"""

