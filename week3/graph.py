"""Graph class for A* algorithm"""
from node import Node


class Graph:
    """Graph structure with nodes and edges"""
    
    def __init__(self):
        """Initialize empty graph"""
        self.nodes = {}
    
    def add_node(self, name, x, y):
        """Add node to graph"""
        if name not in self.nodes:
            node = Node(name, x, y)
            self.nodes[name] = node
            return node
        return None
    
    def get_node(self, name):
        """Get node by name"""
        return self.nodes.get(name)
    
    def add_edge(self, node1_name, node2_name, bidirectional=True):
        """Create edge between two nodes"""
        node1 = self.get_node(node1_name)
        node2 = self.get_node(node2_name)
        
        if node1 and node2 and node1 != node2:
            node1.add_neighbor(node2)
            if bidirectional:
                node2.add_neighbor(node1)
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
        """Get number of nodes"""
        return len(self.nodes)
    
    def get_edge_count(self):
        """Get number of edges"""
        return sum(len(node.neighbors) for node in self.nodes.values())
    
    def clear(self):
        """Clear graph"""
        self.nodes.clear()

