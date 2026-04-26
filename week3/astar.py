"""A* algorithm for pathfinding"""
from heapq import heappush, heappop
""" this heap used for priority queue in A* algorithm"""

"""---"""
class AStarFinder:
    """A* pathfinding algorithm"""
    
    def __init__(self):
        """Initialize A* finder"""
        self.open_list = []
        """ wata au greyany dozrauanta bas lekolynaau le nakre"""
        self.closed_list = set()
        """ wata au nodala explored krauan alerady check kraun"""
        self.exploration_order = []
        """ rebandy grey sardanyrkrau"""
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
        """ g+h"""
        start_node.parent = None
        
        heappush(self.open_list, (start_node.f, id(start_node), start_node))
        """ add start node to the priority queue with f cost and unique id"""
        while self.open_list:
            """ au loop step by step esh daka lo dozyanauy best path"""
            _f_cost, _node_id, current_node = heappop(self.open_list)
            """ takes best node push priority que  au du batala f ,id"""
            
            """ agar already lanau priority queue habu skip bka"""
            if current_node.name in self.closed_list:
            
                continue
            
            self.closed_list.add(current_node.name)
            self.exploration_order.append(current_node.name)
            
            if current_node.name == goal_node.name:
                self.path = self._reconstruct_path(current_node)
                """ rebuild path nmuna agar g,s,b, daygore s,b,g"""
                """ agar current node goal bu buasta la serach krdn"""
                self.total_cost = current_node.g
                return self.path, self.exploration_order, self.total_cost
            
            for neighbor in current_node.neighbors:
                """ for current node check all node connected"""
                if neighbor.name in self.closed_list:
                    """ agar jiranakay haman node ka chuy boy ignoty kba"""
                    continue
                
                edge_cost = current_node.euclidean_distance(neighbor)
                new_g = current_node.g + edge_cost
                """ distance currnt node  to neigbord"""
                
                if new_g < neighbor.g:
                    """ copmare daka auay nwe costy kamter """
                    neighbor.g = new_g
                    """ agar bashtr bu save better cost"""
                    neighbor.parent = current_node
                    if neighbor.h == 0 and neighbor != goal_node:
                        neighbor.calculate_heuristic(goal_node)
                        """ calcluate h dakayn"""
                    neighbor.update_f_cost()
                    heappush(self.open_list, (neighbor.f, id(neighbor), neighbor))
        
        return [], self.exploration_order, self.total_cost
    
    def _reconstruct_path(self, node):
     




     
        path = []
        current = node
        """ start from a goal node"""
        while current is not None:
            path.append(current.name)
            current = current.parent
            """
            G → C → B → S 



 "G", "C", "B", "S"]
            """
        path.reverse()
        return path
    
    def get_exploration_details(self, graph):
      
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
    



"""
 √((2-0)² + (1-0)²)
= √(4 + 1)
= √5 ≈ 2.24

 So:

g(B) = 0 + 2.24 = 2.24
Step 3: From B → C

Distance:

= √((3-2)² + (3-1)²)
= √(1 + 4)
= √5 ≈ 2.24

 Add to previous g:

g(C) = 2.24 + 2.24 = 4.47
wata gn privuse pekau koyan dkay
"""