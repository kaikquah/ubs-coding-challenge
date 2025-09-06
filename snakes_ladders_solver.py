import xml.etree.ElementTree as ET
from flask import Blueprint, request, Response
from collections import deque
from typing import Tuple, Dict, Set, Optional
import logging

logger = logging.getLogger(__name__)
snakes_bp = Blueprint('snakes_ladders', __name__)

class SnakesLaddersSolver:
    def __init__(self, svg_content: str):
        self.parse_board(svg_content)
        self.jumps = {}
        self.parse_jumps(svg_content)
        
    def parse_board(self, svg_content: str):
        """Parse board dimensions from SVG viewBox"""
        root = ET.fromstring(svg_content)
        viewbox = root.get('viewBox').split()
        width = int(viewbox[2])
        height = int(viewbox[3])
        
        # Each square is 32x32 pixels
        self.cols = width // 32
        self.rows = height // 32
        self.total_squares = self.cols * self.rows
        logger.info(f"Board: {self.rows}x{self.cols} = {self.total_squares} squares")
        
    def parse_jumps(self, svg_content: str):
        """Parse snake and ladder jumps from SVG lines"""
        root = ET.fromstring(svg_content)
        
        for line in root.findall('.//line'):
            x1 = int(line.get('x1'))
            y1 = int(line.get('y1'))
            x2 = int(line.get('x2'))
            y2 = int(line.get('y2'))
            
            start_square = self.coord_to_square(x1, y1)
            end_square = self.coord_to_square(x2, y2)
            
            self.jumps[start_square] = end_square
            logger.info(f"Jump: {start_square} -> {end_square}")
    
    def coord_to_square(self, x: int, y: int) -> int:
        """Convert SVG coordinates to square number (1-based, Boustrophedon pattern)"""
        col = x // 32
        row = (self.rows - 1) - (y // 32)
        
        if row % 2 == 0:  # Even rows: left to right
            square = row * self.cols + col + 1
        else:  # Odd rows: right to left
            square = row * self.cols + (self.cols - col)
        
        return square
    
    def move_player(self, pos: int, roll: int, die_type: str) -> Tuple[int, str]:
        """Execute a move and return new position and die type"""
        # Calculate actual movement
        if die_type == 'power':
            actual_move = 2 ** roll  # 2^1=2, 2^2=4, ..., 2^6=64
        else:
            actual_move = roll
        
        new_pos = pos + actual_move
        
        # Handle overshoot
        if new_pos > self.total_squares:
            overshoot = new_pos - self.total_squares
            new_pos = self.total_squares - overshoot
        
        # Apply jump if exists
        if new_pos in self.jumps:
            new_pos = self.jumps[new_pos]
        
        # Determine next die type
        if die_type == 'regular' and roll == 6:
            next_die = 'power'
        elif die_type == 'power' and roll == 1:
            next_die = 'regular'
        else:
            next_die = die_type
        
        return new_pos, next_die
    
    def simulate_game(self, rolls: str) -> Tuple[int, int, Set[int]]:
        """Simulate game with given rolls, return winner and coverage"""
        p1_pos = 0
        p2_pos = 0
        p1_die = 'regular'
        p2_die = 'regular'
        visited = set()
        
        for i, roll in enumerate(rolls):
            roll = int(roll)
            if i % 2 == 0:  # Player 1's turn
                p1_pos, p1_die = self.move_player(p1_pos, roll, p1_die)
                if p1_pos > 0:
                    visited.add(p1_pos)
                if p1_pos == self.total_squares:
                    return 1, len(visited)
            else:  # Player 2's turn
                p2_pos, p2_die = self.move_player(p2_pos, roll, p2_die)
                if p2_pos > 0:
                    visited.add(p2_pos)
                if p2_pos == self.total_squares:
                    return 2, len(visited)
        
        return 0, len(visited)
    
    def find_optimal_solution(self) -> str:
        """Use BFS with pruning to find optimal solution"""
        # State: (p1_pos, p1_die, p2_pos, p2_die, turn, path)
        queue = deque([(0, 'regular', 0, 'regular', 0, "")])
        visited_states = {}
        best_solution = None
        best_score = -1
        
        max_depth = 100  # Limit search depth
        
        while queue:
            p1_pos, p1_die, p2_pos, p2_die, turn, path = queue.popleft()
            
            # Depth limit
            if len(path) > max_depth:
                continue
            
            # State pruning
            state_key = (p1_pos, p1_die, p2_pos, p2_die, turn % 2)
            if state_key in visited_states and visited_states[state_key] <= len(path):
                continue
            visited_states[state_key] = len(path)
            
            # Check win conditions
            if p2_pos == self.total_squares:
                # Player 2 wins (desired)
                winner, coverage = self.simulate_game(path)
                coverage_ratio = coverage / self.total_squares
                
                if coverage_ratio > 0.25:
                    # Calculate score: higher is better, but penalize high coverage
                    score = 25 * (1 - coverage_ratio) if coverage_ratio <= 1 else 0
                    if score > best_score:
                        best_score = score
                        best_solution = path
                        logger.info(f"Found solution: coverage={coverage_ratio:.2%}, score={score}")
                        
                        # Early exit if we found a good solution
                        if 0.25 < coverage_ratio < 0.35:
                            return path
                continue
            
            if p1_pos == self.total_squares:
                # Player 1 wins (undesired)
                continue
            
            # Generate next moves
            current_die = p1_die if turn % 2 == 0 else p2_die
            
            # Prioritize certain rolls based on game state
            if turn % 2 == 1 and p2_pos > self.total_squares * 0.7:
                # Player 2 is close to winning, try optimal moves first
                roll_order = self.get_optimal_rolls(p2_pos, current_die)
            else:
                roll_order = range(1, 7)
            
            for roll in roll_order:
                new_path = path + str(roll)
                
                if turn % 2 == 0:  # Player 1's turn
                    new_p1_pos, new_p1_die = self.move_player(p1_pos, roll, current_die)
                    queue.append((new_p1_pos, new_p1_die, p2_pos, p2_die, turn + 1, new_path))
                else:  # Player 2's turn
                    new_p2_pos, new_p2_die = self.move_player(p2_pos, roll, current_die)
                    queue.append((p1_pos, p1_die, new_p2_pos, new_p2_die, turn + 1, new_path))
        
        # If no optimal solution found, use heuristic approach
        if not best_solution:
            logger.info("Using heuristic solution")
            best_solution = self.generate_heuristic_solution()
        
        return best_solution
    
    def get_optimal_rolls(self, pos: int, die_type: str) -> list:
        """Get rolls ordered by optimality for reaching the end"""
        remaining = self.total_squares - pos
        rolls = []
        
        if die_type == 'regular':
            # Try exact match first, then largest safe move
            for r in [min(remaining, 6), 6, 5, 4, 3, 2, 1]:
                if r not in rolls and 1 <= r <= 6:
                    rolls.append(r)
        else:  # power die
            # Find best power of 2 move
            for r in range(6, 0, -1):
                move = 2 ** r
                if move <= remaining:
                    rolls.append(r)
            for r in range(1, 7):
                if r not in rolls:
                    rolls.append(r)
        
        return rolls
    
    def generate_heuristic_solution(self) -> str:
        """Generate a solution using heuristics"""
        rolls = []
        p1_pos = 0
        p2_pos = 0
        p1_die = 'regular'
        p2_die = 'regular'
        visited = set()
        
        max_moves = 200
        
        while len(rolls) < max_moves and p2_pos != self.total_squares:
            if len(rolls) % 2 == 0:  # Player 1's turn
                # Make suboptimal moves for player 1
                if p1_pos < self.total_squares - 20:
                    roll = 2 if p1_die == 'regular' else 3
                else:
                    roll = 1
                
                p1_pos, p1_die = self.move_player(p1_pos, roll, p1_die)
                if p1_pos > 0:
                    visited.add(p1_pos)
                    
                if p1_pos == self.total_squares:
                    # Player 1 won, restart with different strategy
                    return self.generate_alternative_solution()
                    
            else:  # Player 2's turn
                remaining = self.total_squares - p2_pos
                
                # Choose optimal roll for player 2
                if p2_die == 'regular':
                    if remaining > 10 and len(visited) / self.total_squares < 0.2:
                        roll = 6  # Switch to power die for faster movement
                    elif remaining <= 6:
                        roll = min(remaining, 6)
                    else:
                        roll = min(6, max(3, remaining // 4))
                else:  # power die
                    best_roll = 1
                    for r in range(1, 7):
                        if 2 ** r <= remaining:
                            best_roll = r
                        elif 2 ** r > remaining and remaining > 32:
                            best_roll = r - 1
                            break
                    roll = best_roll
                
                p2_pos, p2_die = self.move_player(p2_pos, roll, p2_die)
                if p2_pos > 0:
                    visited.add(p2_pos)
            
            rolls.append(str(roll))
        
        return ''.join(rolls)
    
    def generate_alternative_solution(self) -> str:
        """Alternative strategy with controlled player 1 movement"""
        rolls = []
        p1_pos = 0
        p2_pos = 0
        p1_die = 'regular'
        p2_die = 'regular'
        
        # Keep player 1 moving slowly while player 2 advances
        while p2_pos != self.total_squares and len(rolls) < 300:
            if len(rolls) % 2 == 0:  # Player 1
                # Use rolls that don't trigger die switch
                roll = 3 if p1_pos < self.total_squares // 2 else 1
                p1_pos, p1_die = self.move_player(p1_pos, roll, p1_die)
                
                if p1_pos == self.total_squares:
                    # Try different approach
                    roll = 2
                    
            else:  # Player 2
                remaining = self.total_squares - p2_pos
                
                if p2_die == 'regular':
                    roll = min(6, remaining) if remaining <= 6 else 5
                else:
                    # Find best power move
                    roll = 1
                    for r in range(6, 0, -1):
                        if 2 ** r <= remaining:
                            roll = r
                            break
                
                p2_pos, p2_die = self.move_player(p2_pos, roll, p2_die)
            
            rolls.append(str(roll))
        
        return ''.join(rolls)

def solve_snakes_ladders(svg_content: str) -> str:
    """Main solving function"""
    try:
        solver = SnakesLaddersSolver(svg_content)
        
        # Try BFS first for optimal solution
        solution = solver.find_optimal_solution()
        
        # Validate solution
        winner, coverage = solver.simulate_game(solution)
        logger.info(f"Final solution: winner=P{winner}, coverage={coverage}/{solver.total_squares}")
        
        # Format as SVG response
        return f'<svg xmlns="http://www.w3.org/2000/svg"><text>{solution}</text></svg>'
    
    except Exception as e:
        logger.error(f"Error solving Snakes & Ladders: {e}")
        # Return a basic fallback solution
        return '<svg xmlns="http://www.w3.org/2000/svg"><text>212121212156</text></svg>'

@snakes_bp.route('/slpu', methods=['POST'])
def snakes_ladders_powerup():
    """Handle Snakes & Ladders Power Up challenge"""
    try:
        # Log request details
        content_type = request.headers.get('Content-Type', '')
        logger.info(f"SLPU endpoint called with content-type: {content_type}")
        
        # Get SVG content
        svg_content = request.get_data(as_text=True)
        logger.info(f"Received SVG content length: {len(svg_content)}")
        
        # Solve the puzzle
        solution = solve_snakes_ladders(svg_content)
        logger.info(f"Generated solution: {solution}")
        
        # Return SVG response with correct content type
        return Response(solution, mimetype='image/svg+xml')
    
    except Exception as e:
        logger.error(f"Error in /slpu endpoint: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Return minimal valid SVG on error
        fallback = '<svg xmlns="http://www.w3.org/2000/svg"><text>111111</text></svg>'
        return Response(fallback, mimetype='image/svg+xml')