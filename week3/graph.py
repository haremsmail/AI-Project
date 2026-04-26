
from node import Node


class Graph:
    
    
    def __init__(self):
        """Initialize empty graph"""
        self.nodes = {}
        """ auayn dictinary data typa"""
    
    def add_node(self, name, x, y):
        """Add node to graph"""
        if name not in self.nodes:
            node = Node(name, x, y)
            self.nodes[name] = node
            return node
        return None
    """ agar nodaka alerady habu return none daka"""
    
    def get_node(self, name):
        """Get node by name"""
        return self.nodes.get(name)
    
    def add_edge(self, node1_name, node2_name, bidirectional=True):
        """ bidarkecatn wata payuandy ba hardu lada darua"""
        """Create edge between two nodes"""
        node1 = self.get_node(node1_name)
        node2 = self.get_node(node2_name)
        
        if node1 and node2 and node1 != node2:
            node1.add_neighbor(node2)
            if bidirectional:
                """ wata hardu node payaudnya pek way haya yan as yakyan"""
                node2.add_neighbor(node1)
                """ edge create success"""
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
        """ agadary sheuny ba clear privuse seraw"""
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
        """Remove all nodes and edges from the graph"""
        self.nodes.clear()

