"""
Node Class - Represents a single node in the graph
Each node contains coordinates, neighbors, and cost values for A* algorithm
"""
import math


class Node:
    """
    Represents a node in the graph for A* pathfinding
    
    Attributes:
        name (str): Label of the node (e.g., 'S', 'A', 'B', 'G')
        x (float): X coordinate on the grid
        y (float): Y coordinate on the grid
        neighbors (list): List of connected nodes (edges)
        g (float): Cost from start node to this node
        h (float): Heuristic cost (Euclidean distance to goal)
        f (float): Total cost f = g + h
        parent (Node): Parent node for path reconstruction
    """
    
    def __init__(self, name, x, y):
        """
        Initialize a node with name and coordinates
        
        Args:
            name (str): Node label
            x (float): X coordinate
            y (float): Y coordinate
        """
        self.name = name
        self.x = x
        self.y = y
        self.neighbors = []  # List of connected nodes
        
        # A* cost values
        self.g = float('inf')  # Cost from start
        self.h = 0             # Heuristic cost to goal
        self.f = float('inf')  # Total cost
        self.parent = None     # For path reconstruction
        
    def add_neighbor(self, neighbor_node):
        """
        Add a neighbor node (create an edge)
        
        Args:
            neighbor_node (Node): Node to connect to
        """
        if neighbor_node not in self.neighbors:
            self.neighbors.append(neighbor_node)
    
    def remove_neighbor(self, neighbor_node):
        """
        Remove a neighbor node (remove an edge)
        
        Args:
            neighbor_node (Node): Node to disconnect
        """
        if neighbor_node in self.neighbors:
            self.neighbors.remove(neighbor_node)
    
    def euclidean_distance(self, other_node):
        """
        Calculate Euclidean distance to another node
        Formula: h(n) = sqrt((x2 - x1)^2 + (y2 - y1)^2)
        
        Args:
            other_node (Node): Target node
            
        Returns:
            float: Euclidean distance
        """
        return math.sqrt((other_node.x - self.x)**2 + (other_node.y - self.y)**2)
    
    def calculate_heuristic(self, goal_node):
        """
        Calculate heuristic value (h) to goal using Euclidean distance
        
        Args:
            goal_node (Node): Goal node
        """
        self.h = self.euclidean_distance(goal_node)
    
    def update_f_cost(self):
        """
        Update the total f cost (f = g + h)
        """
        self.f = self.g + self.h
    
    def reset_costs(self):
        """
        Reset A* costs to initial values (used for multiple runs)
        """
        self.g = float('inf')
        self.h = 0
        self.f = float('inf')
        self.parent = None
    
    def __lt__(self, other):
        """
        Comparison operator for priority queue (compare by f cost)
        Used when nodes are stored in priority queue
        
        Args:
            other (Node): Node to compare with
            
        Returns:
            bool: True if this node's f cost is less than other
        """
        return self.f < other.f
    
    def __eq__(self, other):
        """
        Check if two nodes are the same
        
        Args:
            other (Node): Node to compare with
            
        Returns:
            bool: True if nodes have same name
        """
        if isinstance(other, Node):
            return self.name == other.name
        return False
    
    def __hash__(self):
        """
        Make node hashable for use in sets and dicts
        
        Returns:
            int: Hash of node name
        """
        return hash(self.name)
    
    def __repr__(self):
        """
        String representation of node
        
        Returns:
            str: Node information
        """
        return f"Node({self.name}, ({self.x}, {self.y}), f={self.f:.2f})"
