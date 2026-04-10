"""
Graph Class - Manages collection of nodes and edges for A* algorithm
Provides methods to add/remove nodes, create edges, and access node data
"""
from node import Node


class Graph:
    """
    Represents a graph structure for A* pathfinding
    
    Attributes:
        nodes (dict): Dictionary of nodes with name as key
    """
    
    def __init__(self):
        """Initialize an empty graph"""
        self.nodes = {}  # Dictionary: name -> Node object
    
    def add_node(self, name, x, y):
        """
        Add a new node to the graph
        
        Args:
            name (str): Node label
            x (float): X coordinate
            y (float): Y coordinate
            
        Returns:
            Node: The created node, or None if already exists
        """
        if name not in self.nodes:
            node = Node(name, x, y)
            self.nodes[name] = node
            return node
        return None  # Node already exists
    
    def get_node(self, name):
        """
        Get a node by name
        
        Args:
            name (str): Node label
            
        Returns:
            Node: The node object, or None if doesn't exist
        """
        return self.nodes.get(name)
    
    def add_edge(self, node1_name, node2_name, bidirectional=True):
        """
        Create an edge between two nodes
        
        Args:
            node1_name (str): First node label
            node2_name (str): Second node label
            bidirectional (bool): If True, creates edge both ways (undirected)
                                 If False, creates one-way edge (directed)
            
        Returns:
            bool: True if edge created successfully, False otherwise
        """
        node1 = self.get_node(node1_name)
        node2 = self.get_node(node2_name)
        
        if node1 and node2 and node1 != node2:
            node1.add_neighbor(node2)
            if bidirectional:
                node2.add_neighbor(node1)
            return True
        return False
    
    def remove_edge(self, node1_name, node2_name, bidirectional=True):
        """
        Remove an edge between two nodes
        
        Args:
            node1_name (str): First node label
            node2_name (str): Second node label
            bidirectional (bool): If True, removes edge both ways
            
        Returns:
            bool: True if edge removed successfully, False otherwise
        """
        node1 = self.get_node(node1_name)
        node2 = self.get_node(node2_name)
        
        if node1 and node2:
            node1.remove_neighbor(node2)
            if bidirectional:
                node2.remove_neighbor(node1)
            return True
        return False
    
    def remove_node(self, name):
        """
        Remove a node from the graph (and all its edges)
        
        Args:
            name (str): Node label to remove
            
        Returns:
            bool: True if node removed successfully, False otherwise
        """
        if name in self.nodes:
            node_to_remove = self.nodes[name]
            
            # Remove this node from all neighbors' adjacency lists
            for node in self.nodes.values():
                node.remove_neighbor(node_to_remove)
            
            # Remove the node itself
            del self.nodes[name]
            return True
        return False
    
    def get_all_nodes(self):
        """
        Get all nodes in the graph
        
        Returns:
            list: List of all Node objects
        """
        return list(self.nodes.values())
    
    def reset_all_costs(self):
        """
        Reset A* costs for all nodes (prepare for new search)
        """
        for node in self.nodes.values():
            node.reset_costs()
    
    def get_node_count(self):
        """
        Get the number of nodes in the graph
        
        Returns:
            int: Number of nodes
        """
        return len(self.nodes)
    
    def get_edge_count(self):
        """
        Get the total number of edges in the graph
        
        Returns:
            int: Number of edges
        """
        edge_count = 0
        for node in self.nodes.values():
            edge_count += len(node.neighbors)
        return edge_count // 2  # Divide by 2 for undirected graph
    
    def is_connected(self, start_name, goal_name):
        """
        Check if a path exists between two nodes using BFS
        
        Args:
            start_name (str): Start node label
            goal_name (str): Goal node label
            
        Returns:
            bool: True if path exists, False otherwise
        """
        if start_name not in self.nodes or goal_name not in self.nodes:
            return False
        
        visited = set()
        queue = [self.nodes[start_name]]
        
        while queue:
            node = queue.pop(0)
            if node.name == goal_name:
                return True
            
            if node.name in visited:
                continue
            
            visited.add(node.name)
            for neighbor in node.neighbors:
                if neighbor.name not in visited:
                    queue.append(neighbor)
        
        return False
    
    def clear(self):
        """
        Clear all nodes and edges from the graph
        """
        self.nodes.clear()
    
    def __repr__(self):
        """
        String representation of graph
        
        Returns:
            str: Graph information
        """
        return f"Graph(Nodes: {self.get_node_count()}, Edges: {self.get_edge_count()})"
