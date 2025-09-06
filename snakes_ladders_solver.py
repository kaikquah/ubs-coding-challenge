import xml.etree.ElementTree as ET
from flask import Blueprint, request, Response
from typing import Tuple, Dict, Set, List
import logging
import random

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
    
    def coord_to_square(self, x: int, y: int) -> int:
        """Convert SVG coordinates to square number"""
        col = x // 32
        row = (self.rows - 1) - (y // 32)
        
        if row % 2 == 0:
            square = row * self.cols + col + 1
        else:
            square = row * self.cols + (self.cols - col)
        
        return square
    
    def move_player(self, pos: int, roll: int, die_type: str) -> Tuple[int, str]:
        """Execute a move and return new position and die type"""
        if die_type == 'power':
            actual_move = 2 ** roll
        else:
            actual_move = roll
        
        new_pos = pos + actual_move
        
        # Handle overshoot
        if new_pos > self.total_squares:
            overshoot = new_pos - self.total_squares
            new_pos = self.total_squares - overshoot
        
        # Apply jump
        if new_pos in self.jumps:
            new_pos = self.jumps[new_pos]
        
        # Die type changes
        if die_type == 'regular' and roll == 6:
            next_die = 'power'
        elif die_type == 'power' and roll == 1:
            next_die = 'regular'
        else:
            next_die = die_type
        
        return new_pos, next_die
    
    def simulate_game(self, rolls: str) -> Tuple[int, int]:
        """Simulate game and return winner and coverage"""
        p1_pos = 0
        p2_pos = 0
        p1_die = 'regular'
        p2_die = 'regular'
        visited = set()
        
        for i, roll in enumerate(rolls):
            roll = int(roll)
            if i % 2 == 0:  # Player 1
                p1_pos, p1_die = self.move_player(p1_pos, roll, p1_die)
                if p1_pos > 0:
                    visited.add(p1_pos)
                if p1_pos == self.total_squares:
                    return 1, len(visited)
            else:  # Player 2
                p2_pos, p2_die = self.move_player(p2_pos, roll, p2_die)
                if p2_pos > 0:
                    visited.add(p2_pos)
                if p2_pos == self.total_squares:
                    return 2, len(visited)
        
        return 0, len(visited)
    
    def find_solution(self) -> str:
        """Find a working solution efficiently"""
        best_solution = None
        best_score = -1
        
        # Try multiple random seeds for variety
        for seed in range(10):
            random.seed(seed)
            solution = self.generate_game_sequence()
            
            winner, coverage = self.simulate_game(solution)
            if winner == 2:
                coverage_ratio = coverage / self.total_squares
                if coverage_ratio > 0.25:
                    score = 25 * (1 - coverage_ratio) if coverage_ratio < 1 else 0
                    if score > best_score:
                        best_score = score
                        best_solution = solution
                        
                        # Good enough solution found
                        if 0.26 < coverage_ratio < 0.35:
                            return solution
        
        # If no good solution from random seeds, try deterministic approach
        if not best_solution:
            best_solution = self.generate_deterministic_solution()
        
        return best_solution
    
    def generate_game_sequence(self) -> str:
        """Generate a single game sequence"""
        rolls = []
        p1_pos = 0
        p2_pos = 0
        p1_die = 'regular'
        p2_die = 'regular'
        
        max_moves = min(400, self.total_squares * 2)
        
        while len(rolls) < max_moves and p2_pos != self.total_squares:
            if len(rolls) % 2 == 0:  # Player 1
                # P1 strategy: mostly small moves
                if p1_pos < self.total_squares * 0.8:
                    if p1_die == 'regular':
                        roll = random.choice([1, 2, 2, 3, 3, 4])
                    else:
                        roll = random.choice([1, 1, 2, 3])
                else:
                    # Near end, be more careful
                    roll = 1
                
                # Check if this would make P1 win
                test_pos, _ = self.move_player(p1_pos, roll, p1_die)
                if test_pos == self.total_squares:
                    # Avoid P1 winning
                    roll = 1 if roll != 1 else 2
                
                p1_pos, p1_die = self.move_player(p1_pos, roll, p1_die)
                
            else:  # Player 2
                remaining = self.total_squares - p2_pos
                
                if p2_die == 'regular':
                    if remaining <= 6:
                        roll = remaining
                    elif remaining < 20:
                        roll = random.choice([3, 4, 5])
                    elif remaining < 50:
                        roll = random.choice([4, 5, 5, 6])
                    else:
                        roll = random.choice([5, 6, 6])  # Favor getting power die
                else:  # power die
                    # Calculate best power move
                    best_roll = 1
                    for r in range(6, 0, -1):
                        move = 2 ** r
                        if move <= remaining:
                            best_roll = r
                            break
                        elif move < remaining * 1.5 and remaining > 64:
                            # Can overshoot and bounce back acceptably
                            best_roll = r
                            break
                    
                    roll = best_roll
                
                p2_pos, p2_die = self.move_player(p2_pos, roll, p2_die)
            
            rolls.append(str(roll))
        
        return ''.join(rolls)
    
    def generate_deterministic_solution(self) -> str:
        """Deterministic fallback solution"""
        rolls = []
        p1_pos = 0
        p2_pos = 0
        p1_die = 'regular'
        p2_die = 'regular'
        
        # Simple deterministic pattern
        p1_pattern = [2, 1, 3, 1, 2, 3, 1, 2]
        p2_pattern = [5, 4, 6, 3, 5, 4, 5, 6]
        pattern_idx = 0
        
        while p2_pos != self.total_squares and len(rolls) < 500:
            if len(rolls) % 2 == 0:  # Player 1
                roll = p1_pattern[pattern_idx % len(p1_pattern)]
                
                # Safety check
                test_pos, _ = self.move_player(p1_pos, roll, p1_die)
                if test_pos == self.total_squares:
                    roll = 1
                
                p1_pos, p1_die = self.move_player(p1_pos, roll, p1_die)
                
            else:  # Player 2
                remaining = self.total_squares - p2_pos
                
                if p2_die == 'regular':
                    if remaining <= 6:
                        roll = remaining
                    else:
                        roll = p2_pattern[pattern_idx % len(p2_pattern)]
                else:  # power die
                    # Simple power die logic
                    if remaining >= 64:
                        roll = 6
                    elif remaining >= 32:
                        roll = 5
                    elif remaining >= 16:
                        roll = 4
                    elif remaining >= 8:
                        roll = 3
                    elif remaining >= 4:
                        roll = 2
                    else:
                        roll = 1
                
                p2_pos, p2_die = self.move_player(p2_pos, roll, p2_die)
                pattern_idx += 1
            
            rolls.append(str(roll))
        
        return ''.join(rolls)

def solve_snakes_ladders(svg_content: str) -> str:
    """Main solving function"""
    try:
        solver = SnakesLaddersSolver(svg_content)
        
        # Find solution
        solution = solver.find_solution()
        
        # Validate
        winner, coverage = solver.simulate_game(solution)
        logger.info(f"Solution: winner=P{winner}, coverage={coverage}/{solver.total_squares}, len={len(solution)}")
        
        # If still invalid, use emergency fallback
        if winner != 2:
            logger.warning("Using emergency fallback")
            solution = "21212121212156"  # Minimal fallback
        
        return f'<svg xmlns="http://www.w3.org/2000/svg"><text>{solution}</text></svg>'
    
    except Exception as e:
        logger.error(f"Error solving: {e}")
        return '<svg xmlns="http://www.w3.org/2000/svg"><text>212121256</text></svg>'

@snakes_bp.route('/slpu', methods=['POST'])
def snakes_ladders_powerup():
    """Handle Snakes & Ladders Power Up challenge"""
    try:
        svg_content = request.get_data(as_text=True)
        solution = solve_snakes_ladders(svg_content)
        return Response(solution, mimetype='image/svg+xml')
    except Exception as e:
        logger.error(f"Error in /slpu: {e}")
        fallback = '<svg xmlns="http://www.w3.org/2000/svg"><text>111111</text></svg>'
        return Response(fallback, mimetype='image/svg+xml')