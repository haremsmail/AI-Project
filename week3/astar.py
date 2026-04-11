"""A* algorithm for pathfinding"""
from heapq import heappush, heappop


class AStarFinder:
    """A* pathfinding algorithm"""
    
    def __init__(self):
        """Initialize A* finder"""
        self.open_list = []
        self.closed_list = set()
        self.exploration_order = []
        self.path = []
        self.total_cost = 0
    
    def find_path(self, start_node, goal_node):
        """Find shortest path using A* algorithm"""
        self.open_list = []
        self.closed_list = set()
        self.exploration_order = []
        self.path = []
        self.total_cost = 0
        
        start_node.g = 0
        start_node.calculate_heuristic(goal_node)
        start_node.update_f_cost()
        start_node.parent = None
        
        heappush(self.open_list, (start_node.f, id(start_node), start_node))
        
        while self.open_list:
            _, _, current_node = heappop(self.open_list)
            
            if current_node.name in self.closed_list:
                continue
            
            self.closed_list.add(current_node.name)
            self.exploration_order.append(current_node.name)
            
            if current_node.name == goal_node.name:
                self.path = self._reconstruct_path(current_node)
                self.total_cost = current_node.g
                return self.path, self.exploration_order, self.total_cost
            
            for neighbor in current_node.neighbors:
                if neighbor.name in self.closed_list:
                    continue
                
                edge_cost = current_node.euclidean_distance(neighbor)
                new_g = current_node.g + edge_cost
                
                if new_g < neighbor.g:
                    neighbor.g = new_g
                    neighbor.parent = current_node
                    if neighbor.h == 0 and neighbor != goal_node:
                        neighbor.calculate_heuristic(goal_node)
                    neighbor.update_f_cost()
                    heappush(self.open_list, (neighbor.f, id(neighbor), neighbor))
        
        return [], self.exploration_order, self.total_cost
    
    def _reconstruct_path(self, node):
        """Reconstruct path from start to goal"""
        path = []
        current = node
        while current is not None:
            path.append(current.name)
            current = current.parent
        path.reverse()
        return path
    
    def get_exploration_details(self, graph):
        """Get details of each explored node"""
        details = []
        for node_name in self.exploration_order:
            node = graph.get_node(node_name)
            if node:
                details.append({
                    'node': node.name,
                    'x': node.x,
                    'y': node.y,
                    'g': round(node.g, 2),
                    'h': round(node.h, 2),
                    'f': round(node.f, 2)
                })
        return details
    
