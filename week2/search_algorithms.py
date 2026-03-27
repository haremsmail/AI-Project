"""
Search algorithms: DFS, BFS, and Best-First Search
Used for both 8-puzzle and vacuum problems
"""

from collections import deque
import heapq
from typing import List, Tuple, Dict, Set, Optional


class SearchAlgorithm:
    """Base class for search algorithms"""
    
    def __init__(self, initial_state, goal_test, get_neighbors, get_cost):
        self.initial_state = initial_state
        self.goal_test = goal_test
        self.get_neighbors = get_neighbors
        self.get_cost = get_cost
        self.solution = []
        self.total_cost = 0
        self.explored = 0
        
    def solve(self):
        """Override in subclasses"""
        raise NotImplementedError

    def _state_id(self, state):
        """Convert a state to a hashable identifier."""
        if hasattr(state, '__hash__') and not isinstance(state, (list, tuple)):
            return state
        if isinstance(state, (list, tuple)):
            return tuple(tuple(row) if isinstance(row, list) else row for row in state)
        return state

    def _reconstruct_solution(self, goal_id, parents, states):
        """Rebuild path as [(state, move_cost), ...] from parent links."""
        path = []
        current_id = goal_id

        while parents[current_id] is not None:
            parent_id, move_cost = parents[current_id]
            path.append((states[current_id], move_cost))
            current_id = parent_id

        path.reverse()
        return path


class DepthFirstSearch(SearchAlgorithm):
    """Depth-First Search implementation"""
    
    def solve(self):
        """Solve using DFS"""
        initial_id = self._state_id(self.initial_state)
        stack = [self.initial_state]
        visited = {initial_id}
        parents = {initial_id: None}
        states = {initial_id: self.initial_state}
        
        while stack:
            current_state = stack.pop()
            current_id = self._state_id(current_state)
            self.explored += 1
            
            if self.goal_test(current_state):
                self.solution = self._reconstruct_solution(current_id, parents, states)
                self.total_cost = sum(cost for _, cost in self.solution)
                return True
            
            # Add neighbors to stack in reverse order for proper DFS
            neighbors = self.get_neighbors(current_state)
            for neighbor_state, cost in reversed(neighbors):
                state_id = self._state_id(neighbor_state)
                if state_id not in visited:
                    visited.add(state_id)
                    parents[state_id] = (current_id, cost)
                    states[state_id] = neighbor_state
                    stack.append(neighbor_state)
        
        return False


class BreadthFirstSearch(SearchAlgorithm):
    """Breadth-First Search implementation"""
    
    def solve(self):
        """Solve using BFS"""
        initial_id = self._state_id(self.initial_state)
        queue = deque([self.initial_state])
        visited = {initial_id}
        parents = {initial_id: None}
        states = {initial_id: self.initial_state}
        
        while queue:
            current_state = queue.popleft()
            current_id = self._state_id(current_state)
            self.explored += 1
            
            if self.goal_test(current_state):
                self.solution = self._reconstruct_solution(current_id, parents, states)
                self.total_cost = sum(cost for _, cost in self.solution)
                return True
            
            neighbors = self.get_neighbors(current_state)
            for neighbor_state, cost in neighbors:
                state_id = self._state_id(neighbor_state)
                if state_id not in visited:
                    visited.add(state_id)
                    parents[state_id] = (current_id, cost)
                    states[state_id] = neighbor_state
                    queue.append(neighbor_state)
        
        return False


class BestFirstSearch(SearchAlgorithm):
    """Best-First Search implementation with heuristic"""
    
    def __init__(self, initial_state, goal_test, get_neighbors, get_cost, heuristic):
        super().__init__(initial_state, goal_test, get_neighbors, get_cost)
        self.heuristic = heuristic
    
    def solve(self):
        """Solve using Best-First Search"""
        # Priority queue: (heuristic_value, counter, state)
        counter = 0
        initial_h = self.heuristic(self.initial_state)
        heap = [(initial_h, counter, self.initial_state)]
        initial_id = self._state_id(self.initial_state)
        visited = {initial_id}
        parents = {initial_id: None}
        states = {initial_id: self.initial_state}
        counter += 1
        
        while heap:
            _, _, current_state = heapq.heappop(heap)
            current_id = self._state_id(current_state)
            self.explored += 1
            
            if self.goal_test(current_state):
                self.solution = self._reconstruct_solution(current_id, parents, states)
                self.total_cost = sum(cost for _, cost in self.solution)
                return True
            
            neighbors = self.get_neighbors(current_state)
            for neighbor_state, cost in neighbors:
                state_id = self._state_id(neighbor_state)
                if state_id not in visited:
                    visited.add(state_id)
                    parents[state_id] = (current_id, cost)
                    states[state_id] = neighbor_state
                    h_value = self.heuristic(neighbor_state)
                    heapq.heappush(heap, (h_value, counter, neighbor_state))
                    counter += 1
        
        return False
