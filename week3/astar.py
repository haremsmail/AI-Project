"""
A* Algorithm Implementation - Core pathfinding algorithm
Manually implemented WITHOUT external libraries
Uses Euclidean distance as heuristic
"""
from heapq import heappush, heappop


class AStarFinder:
    """
    A* pathfinding algorithm implementation
    
    Features:
        - Uses Euclidean distance as heuristic
        - Maintains open list (priority queue) and closed list
        - Tracks exploration steps for visualization
        - Reconstructs path using parent pointers
    """
    
    def __init__(self):
        """Initialize A* finder"""
        self.open_list = []      # Priority queue of nodes to explore
        self.closed_list = set()  # Set of explored nodes
        self.exploration_order = []  # Track order of exploration for visualization
        self.path = []           # Final path from start to goal
        self.total_cost = 0      # Total cost of final path
    
    def find_path(self, start_node, goal_node):
        """
        Find shortest path from start to goal using A* algorithm
        
        Algorithm Steps:
            1. Initialize open list with start node (g=0)
            2. While open list is not empty:
                a. Get node with lowest f cost from open list
                b. If it's the goal, reconstruct and return path
                c. Otherwise, expand neighbors:
                    - Skip if in closed list
                    - Calculate g cost (distance from start)
                    - Calculate h cost (Euclidean distance to goal)
                    - Calculate f cost (g + h)
                    - If neighbor has better g cost, update it
            3. Return path if found, empty list if not
        
        Args:
            start_node (Node): Starting node
            goal_node (Node): Goal/target node
            
        Returns:
            tuple: (path, exploration_order, total_cost)
                - path: List of nodes from start to goal
                - exploration_order: List of nodes in exploration order
                - total_cost: Total g cost of path
        """
        # Reset data structures for new search
        self.open_list = []
        self.closed_list = set()
        self.exploration_order = []
        self.path = []
        self.total_cost = 0
        
        # Initialize start node
        start_node.g = 0
        start_node.calculate_heuristic(goal_node)
        start_node.update_f_cost()
        start_node.parent = None
        
        # Add start node to open list
        # Using tuple (f_cost, unique_id, node) for heap stability
        heappush(self.open_list, (start_node.f, id(start_node), start_node))
        
        # Main A* loop
        while self.open_list:
            # Get node with lowest f cost (greedy choice)
            _, _, current_node = heappop(self.open_list)
            
            # Skip if already explored
            if current_node.name in self.closed_list:
                continue
            
            # Mark as explored
            self.closed_list.add(current_node.name)
            self.exploration_order.append(current_node.name)
            
            # Goal found!
            if current_node.name == goal_node.name:
                self.path = self._reconstruct_path(current_node)
                self.total_cost = current_node.g
                return self.path, self.exploration_order, self.total_cost
            
            # Expand neighbors (open up a node)
            for neighbor in current_node.neighbors:
                # Skip if already explored
                if neighbor.name in self.closed_list:
                    continue
                
                # Calculate new g cost (g = parent's g + edge distance)
                # Edge distance is Euclidean distance between nodes
                edge_cost = current_node.euclidean_distance(neighbor)
                new_g = current_node.g + edge_cost
                
                # If this is a better path to neighbor, update it
                if new_g < neighbor.g:
                    neighbor.g = new_g
                    neighbor.parent = current_node
                    
                    # Calculate heuristic if not already calculated
                    if neighbor.h == 0 and neighbor != goal_node:
                        neighbor.calculate_heuristic(goal_node)
                    
                    # Update f cost
                    neighbor.update_f_cost()
                    
                    # Add to open list for exploration
                    heappush(self.open_list, (neighbor.f, id(neighbor), neighbor))
        
        # No path found
        return [], self.exploration_order, self.total_cost
    
    def _reconstruct_path(self, node):
        """
        Reconstruct the path from start to current node using parent pointers
        
        Algorithm:
            1. Start from current node
            2. Follow parent pointers back to start (parent = None)
            3. Reverse to get path from start to goal
        
        Args:
            node (Node): Final node (usually goal node)
            
        Returns:
            list: Path as list of node names
        """
        path = []
        current = node
        
        while current is not None:
            path.append(current.name)
            current = current.parent
        
        path.reverse()
        return path
    
    def get_exploration_details(self, graph):
        """
        Get detailed information about each step of exploration
        
        Returns:
            list: List of dictionaries with step information:
                [{'node': 'A', 'g': 1.5, 'h': 2.3, 'f': 3.8}, ...]
        """
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
    
    def get_path_string(self):
        """
        Get path as a formatted string
        
        Returns:
            str: Path formatted as "S → A → C → G"
        """
        if not self.path:
            return "No path found"
        
        return " → ".join(self.path)
    
    def is_optimal(self):
        """
        Check if a path was found
        
        Returns:
            bool: True if path exists, False otherwise
        """
        return len(self.path) > 0
    
    def get_stats(self):
        """
        Get statistics about the search
        
        Returns:
            dict: Statistics dictionary
        """
        return {
            'nodes_explored': len(self.closed_list),
            'path_length': len(self.path),
            'total_cost': round(self.total_cost, 2),
            'path_string': self.get_path_string(),
            'success': self.is_optimal()
        }
