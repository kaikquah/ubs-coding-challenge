from flask import Blueprint, request, jsonify
import logging
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass, field
from collections import deque
import math
from enum import Enum

logger = logging.getLogger(__name__)

# Create blueprint for the fog of wall endpoint
fog_bp = Blueprint('fog_of_wall', __name__)

class Strategy(Enum):
    """Different exploration strategies"""
    GRID_SCAN = 1  # Systematic grid scanning
    WALL_FOLLOW = 2  # Follow discovered walls
    FRONTIER_EXPLORE = 3  # Explore boundaries
    FILL_GAPS = 4  # Fill remaining gaps

@dataclass
class Crow:
    """Represents a crow agent in the maze"""
    id: str
    x: int
    y: int
    last_scan_position: Optional[Tuple[int, int]] = None
    assigned_region: Optional[int] = None
    
@dataclass
class GameState:
    """Maintains the current state of the game"""
    challenger_id: str
    game_id: str
    crows: Dict[str, Crow]
    num_walls: int
    grid_size: int
    discovered_walls: Set[Tuple[int, int]]
    explored_cells: Set[Tuple[int, int]]
    move_count: int
    scan_coverage: Set[Tuple[int, int]]  # Cells covered by scans
    scan_centers: Set[Tuple[int, int]]  # Positions from which we've scanned
    strategy: Strategy = Strategy.GRID_SCAN
    grid_scan_targets: List[Tuple[int, int]] = field(default_factory=list)
    wall_frontier: Set[Tuple[int, int]] = field(default_factory=set)
    
class FogOfWallSolver:
    def __init__(self):
        self.game_states: Dict[str, GameState] = {}
        
    def process_request(self, data: dict) -> dict:
        """Main entry point for processing requests"""
        challenger_id = data.get('challenger_id')
        game_id = data.get('game_id')
        
        # Check if this is an initial request
        if 'test_case' in data:
            return self.handle_initial_request(data)
        
        # Otherwise, it's a move/scan result
        state_key = f"{challenger_id}_{game_id}"
        if state_key not in self.game_states:
            logger.error(f"Unknown game state: {state_key}")
            return {"error": "Unknown game state"}
        
        state = self.game_states[state_key]
        
        # Process previous action result
        if 'previous_action' in data:
            self.process_action_result(state, data['previous_action'])
        
        # Check if we should change strategy
        self.update_strategy(state)
        
        # Decide next action
        return self.decide_next_action(state)
    
    def handle_initial_request(self, data: dict) -> dict:
        """Initialize game state and return first action"""
        challenger_id = data['challenger_id']
        game_id = data['game_id']
        test_case = data['test_case']
        
        # Create crows
        crows = {}
        for crow_data in test_case['crows']:
            crow = Crow(
                id=crow_data['id'],
                x=crow_data['x'],
                y=crow_data['y']
            )
            crows[crow.id] = crow
        
        # Initialize game state
        state = GameState(
            challenger_id=challenger_id,
            game_id=game_id,
            crows=crows,
            num_walls=test_case['num_of_walls'],
            grid_size=test_case['length_of_grid'],
            discovered_walls=set(),
            explored_cells=set(),
            move_count=0,
            scan_coverage=set(),
            scan_centers=set()
        )
        
        # Generate initial scan targets for grid strategy
        state.grid_scan_targets = self.generate_grid_scan_targets(state)
        
        # Assign regions to crows if multiple
        if len(crows) > 1:
            self.assign_crow_regions(state)
        
        # Store state
        state_key = f"{challenger_id}_{game_id}"
        self.game_states[state_key] = state
        
        # Start with scanning from the first crow
        first_crow = list(crows.values())[0]
        return {
            "challenger_id": challenger_id,
            "game_id": game_id,
            "crow_id": first_crow.id,
            "action_type": "scan"
        }
    
    def generate_grid_scan_targets(self, state: GameState) -> List[Tuple[int, int]]:
        """Generate optimal scan positions for grid coverage"""
        targets = []
        
        # Use 4-spacing for optimal coverage (5x5 scan with 1 overlap)
        # This ensures no gaps in coverage
        spacing = 4
        
        # Start from position (2,2) to maximize coverage
        for y in range(2, state.grid_size, spacing):
            for x in range(2, state.grid_size, spacing):
                targets.append((x, y))
        
        # Add edge positions if needed
        # Top and bottom edges
        for x in range(2, state.grid_size, spacing):
            if 2 not in [y for _, y in targets if _ == x]:
                targets.append((x, min(2, state.grid_size - 1)))
            if state.grid_size - 3 not in [y for _, y in targets if _ == x]:
                targets.append((x, max(state.grid_size - 3, 0)))
        
        # Left and right edges
        for y in range(2, state.grid_size, spacing):
            if 2 not in [x for x, _ in targets if _ == y]:
                targets.append((min(2, state.grid_size - 1), y))
            if state.grid_size - 3 not in [x for x, _ in targets if _ == y]:
                targets.append((max(state.grid_size - 3, 0), y))
        
        return targets
    
    def assign_crow_regions(self, state: GameState):
        """Assign different regions to multiple crows"""
        num_crows = len(state.crows)
        if num_crows == 1:
            return
        
        # Divide grid into regions
        crows_list = list(state.crows.values())
        
        if num_crows == 2:
            # Split horizontally
            crows_list[0].assigned_region = 0  # Top half
            crows_list[1].assigned_region = 1  # Bottom half
        elif num_crows == 3:
            # Split into thirds
            for i, crow in enumerate(crows_list):
                crow.assigned_region = i
    
    def process_action_result(self, state: GameState, action: dict):
        """Process the result of a previous action"""
        state.move_count += 1
        
        if action['your_action'] == 'move':
            # Update crow position
            crow_id = action['crow_id']
            new_pos = action['move_result']
            state.crows[crow_id].x = new_pos[0]
            state.crows[crow_id].y = new_pos[1]
            state.explored_cells.add(tuple(new_pos))
            
        elif action['your_action'] == 'scan':
            # Process scan results
            crow_id = action['crow_id']
            crow = state.crows[crow_id]
            scan_result = action['scan_result']
            
            # Mark this position as scanned from
            state.scan_centers.add((crow.x, crow.y))
            crow.last_scan_position = (crow.x, crow.y)
            
            # Process the 5x5 scan grid
            for dy in range(5):
                for dx in range(5):
                    # Calculate actual position
                    actual_x = crow.x + dx - 2
                    actual_y = crow.y + dy - 2
                    
                    # Mark as covered by scan
                    if 0 <= actual_x < state.grid_size and 0 <= actual_y < state.grid_size:
                        state.scan_coverage.add((actual_x, actual_y))
                    
                    cell = scan_result[dy][dx]
                    
                    if cell == 'W':
                        # Found a wall
                        state.discovered_walls.add((actual_x, actual_y))
                        # Add adjacent cells to wall frontier
                        for adj_x, adj_y in self.get_adjacent_cells(actual_x, actual_y):
                            if (0 <= adj_x < state.grid_size and 
                                0 <= adj_y < state.grid_size and
                                (adj_x, adj_y) not in state.scan_coverage):
                                state.wall_frontier.add((adj_x, adj_y))
                    elif cell == '_':
                        # Empty cell
                        state.explored_cells.add((actual_x, actual_y))
    
    def get_adjacent_cells(self, x: int, y: int) -> List[Tuple[int, int]]:
        """Get adjacent cells (4-connected)"""
        return [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
    
    def update_strategy(self, state: GameState):
        """Update exploration strategy based on current state"""
        coverage_ratio = len(state.scan_coverage) / (state.grid_size * state.grid_size)
        walls_found_ratio = len(state.discovered_walls) / max(1, state.num_walls)
        
        # If we've found most walls but not all, switch to gap filling
        if walls_found_ratio > 0.9 and walls_found_ratio < 1.0:
            state.strategy = Strategy.FILL_GAPS
        # If we have good coverage but missing walls, follow wall patterns
        elif coverage_ratio > 0.7 and walls_found_ratio < 0.8:
            state.strategy = Strategy.WALL_FOLLOW
        # If we have wall frontier to explore
        elif len(state.wall_frontier) > 0 and coverage_ratio > 0.3:
            state.strategy = Strategy.FRONTIER_EXPLORE
        # Otherwise continue grid scanning
        else:
            state.strategy = Strategy.GRID_SCAN
    
    def decide_next_action(self, state: GameState) -> dict:
        """Decide the next action based on current state and strategy"""
        
        # Check if we've found all walls
        if len(state.discovered_walls) >= state.num_walls:
            return self.submit_solution(state)
        
        # Check if we've exceeded move limit
        max_moves = min(state.grid_size * state.grid_size, 1000)  # Cap at 1000 moves
        if state.move_count >= max_moves - 10:
            return self.submit_solution(state)
        
        # Check coverage and consider submitting if we have good coverage
        coverage_ratio = len(state.scan_coverage) / (state.grid_size * state.grid_size)
        if coverage_ratio > 0.95:  # 95% coverage
            return self.submit_solution(state)
        
        # Choose action based on strategy
        if state.strategy == Strategy.GRID_SCAN:
            action = self.grid_scan_action(state)
        elif state.strategy == Strategy.WALL_FOLLOW:
            action = self.wall_follow_action(state)
        elif state.strategy == Strategy.FRONTIER_EXPLORE:
            action = self.frontier_explore_action(state)
        else:  # FILL_GAPS
            action = self.fill_gaps_action(state)
        
        if action:
            return action
        
        # Fallback: try any strategy that works
        for strategy_func in [self.grid_scan_action, self.frontier_explore_action, 
                             self.wall_follow_action, self.fill_gaps_action]:
            action = strategy_func(state)
            if action:
                return action
        
        # If still no action, submit what we have
        return self.submit_solution(state)
    
    def grid_scan_action(self, state: GameState) -> Optional[dict]:
        """Execute grid scanning strategy"""
        # First, scan from current positions if beneficial
        for crow_id, crow in state.crows.items():
            if (crow.x, crow.y) not in state.scan_centers:
                if self.count_uncovered_in_range(state, crow.x, crow.y) >= 5:
                    return self.create_scan_action(state, crow_id)
        
        # Find nearest uncovered grid target for each crow
        best_action = None
        min_distance = float('inf')
        
        for target in state.grid_scan_targets:
            if target in state.scan_centers:
                continue
            
            # Check if target would reveal new areas
            if self.count_uncovered_in_range(state, target[0], target[1]) < 5:
                continue
            
            # Find closest crow
            for crow_id, crow in state.crows.items():
                # Check region assignment
                if crow.assigned_region is not None:
                    # Check if target is in crow's region
                    region_size = state.grid_size // len(state.crows)
                    region_start = crow.assigned_region * region_size
                    region_end = region_start + region_size
                    if not (region_start <= target[1] < region_end):
                        continue
                
                distance = abs(crow.x - target[0]) + abs(crow.y - target[1])
                
                if distance == 0:
                    # Already at target, scan
                    return self.create_scan_action(state, crow_id)
                elif distance < min_distance:
                    # Find path to target
                    path = self.find_path(state, (crow.x, crow.y), target)
                    if path and len(path) > 0:
                        min_distance = distance
                        best_action = self.create_move_action(state, crow_id, path[0])
        
        return best_action
    
    def wall_follow_action(self, state: GameState) -> Optional[dict]:
        """Follow walls to discover patterns"""
        # Find crows near walls
        for crow_id, crow in state.crows.items():
            # Check if crow is adjacent to a wall
            adjacent_walls = []
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                if (crow.x + dx, crow.y + dy) in state.discovered_walls:
                    adjacent_walls.append((dx, dy))
            
            if adjacent_walls:
                # Try to move along the wall
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    new_x, new_y = crow.x + dx, crow.y + dy
                    
                    # Check if position is valid and unexplored
                    if (self.is_valid_position(state, new_x, new_y) and
                        (new_x, new_y) not in state.scan_centers):
                        
                        # Check if adjacent to wall
                        wall_adjacent = False
                        for wx, wy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            if (new_x + wx, new_y + wy) in state.discovered_walls:
                                wall_adjacent = True
                                break
                        
                        if wall_adjacent:
                            # Move there
                            direction = self.get_direction(dx, dy)
                            if direction:
                                return self.create_move_action(state, crow_id, direction)
        
        return None
    
    def frontier_explore_action(self, state: GameState) -> Optional[dict]:
        """Explore wall frontier areas"""
        if not state.wall_frontier:
            return None
        
        # Clean frontier (remove already scanned areas)
        state.wall_frontier = {pos for pos in state.wall_frontier 
                              if pos not in state.scan_coverage}
        
        if not state.wall_frontier:
            return None
        
        # Find closest frontier position for each crow
        best_action = None
        min_distance = float('inf')
        
        for frontier_pos in state.wall_frontier:
            for crow_id, crow in state.crows.items():
                distance = abs(crow.x - frontier_pos[0]) + abs(crow.y - frontier_pos[1])
                
                if distance <= 2:  # Within scan range
                    # Scan from current position
                    if (crow.x, crow.y) not in state.scan_centers:
                        return self.create_scan_action(state, crow_id)
                elif distance < min_distance:
                    # Move towards frontier
                    path = self.find_path(state, (crow.x, crow.y), frontier_pos)
                    if path and len(path) > 0:
                        min_distance = distance
                        best_action = self.create_move_action(state, crow_id, path[0])
        
        return best_action
    
    def fill_gaps_action(self, state: GameState) -> Optional[dict]:
        """Fill remaining gaps in coverage"""
        # Find all uncovered cells
        uncovered = []
        for y in range(state.grid_size):
            for x in range(state.grid_size):
                if (x, y) not in state.scan_coverage:
                    uncovered.append((x, y))
        
        if not uncovered:
            return None
        
        # Find best position to scan from
        best_position = None
        max_coverage = 0
        
        for y in range(state.grid_size):
            for x in range(state.grid_size):
                if (x, y) in state.discovered_walls:
                    continue
                
                coverage = self.count_uncovered_in_range(state, x, y)
                if coverage > max_coverage:
                    max_coverage = coverage
                    best_position = (x, y)
        
        if best_position and max_coverage >= 3:
            # Find closest crow
            closest_crow = None
            min_distance = float('inf')
            
            for crow_id, crow in state.crows.items():
                distance = abs(crow.x - best_position[0]) + abs(crow.y - best_position[1])
                if distance < min_distance:
                    min_distance = distance
                    closest_crow = crow_id
            
            if closest_crow:
                crow = state.crows[closest_crow]
                if (crow.x, crow.y) == best_position:
                    return self.create_scan_action(state, closest_crow)
                else:
                    path = self.find_path(state, (crow.x, crow.y), best_position)
                    if path and len(path) > 0:
                        return self.create_move_action(state, closest_crow, path[0])
        
        return None
    
    def count_uncovered_in_range(self, state: GameState, x: int, y: int) -> int:
        """Count uncovered cells in scan range from position"""
        count = 0
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                check_x, check_y = x + dx, y + dy
                if (0 <= check_x < state.grid_size and 
                    0 <= check_y < state.grid_size and
                    (check_x, check_y) not in state.scan_coverage):
                    count += 1
        return count
    
    def find_path(self, state: GameState, start: Tuple[int, int], 
                  target: Tuple[int, int]) -> Optional[List[str]]:
        """BFS pathfinding"""
        if start == target:
            return []
        
        queue = deque([(start[0], start[1], [])])
        visited = {start}
        
        while queue:
            x, y, path = queue.popleft()
            
            for direction, (dx, dy) in [('N', (0, -1)), ('S', (0, 1)), 
                                       ('E', (1, 0)), ('W', (-1, 0))]:
                new_x, new_y = x + dx, y + dy
                
                if (new_x, new_y) == target:
                    return path + [direction]
                
                if ((new_x, new_y) not in visited and 
                    self.is_valid_position(state, new_x, new_y)):
                    visited.add((new_x, new_y))
                    queue.append((new_x, new_y, path + [direction]))
        
        return None
    
    def is_valid_position(self, state: GameState, x: int, y: int) -> bool:
        """Check if a position is valid and not a wall"""
        if not (0 <= x < state.grid_size and 0 <= y < state.grid_size):
            return False
        return (x, y) not in state.discovered_walls
    
    def get_direction(self, dx: int, dy: int) -> Optional[str]:
        """Convert delta to direction"""
        if dx == 0 and dy == -1:
            return 'N'
        elif dx == 0 and dy == 1:
            return 'S'
        elif dx == 1 and dy == 0:
            return 'E'
        elif dx == -1 and dy == 0:
            return 'W'
        return None
    
    def create_scan_action(self, state: GameState, crow_id: str) -> dict:
        """Create a scan action"""
        return {
            "challenger_id": state.challenger_id,
            "game_id": state.game_id,
            "crow_id": crow_id,
            "action_type": "scan"
        }
    
    def create_move_action(self, state: GameState, crow_id: str, direction: str) -> dict:
        """Create a move action"""
        return {
            "challenger_id": state.challenger_id,
            "game_id": state.game_id,
            "crow_id": crow_id,
            "action_type": "move",
            "direction": direction
        }
    
    def submit_solution(self, state: GameState) -> dict:
        """Submit the discovered walls"""
        # Convert wall positions to required format
        wall_positions = [f"{x}-{y}" for x, y in sorted(state.discovered_walls)]
        
        coverage_ratio = len(state.scan_coverage) / (state.grid_size * state.grid_size)
        logger.info(f"Submitting {len(wall_positions)}/{state.num_walls} walls, "
                   f"coverage: {coverage_ratio:.2%}, moves: {state.move_count}")
        
        return {
            "challenger_id": state.challenger_id,
            "game_id": state.game_id,
            "action_type": "submit",
            "submission": wall_positions
        }

# Global solver instance
solver = FogOfWallSolver()

@fog_bp.route('/fog-of-wall', methods=['POST'])
def fog_of_wall():
    """Main endpoint for the Fog of Wall challenge"""
    try:
        data = request.get_json()
        logger.info(f"Fog of Wall request received")
        
        result = solver.process_request(data)
        logger.info(f"Fog of Wall response: action={result.get('action_type')}")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in fog-of-wall: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500