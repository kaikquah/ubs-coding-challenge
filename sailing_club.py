"""
Sailing Club Booking Optimizer
Solves boat booking interval merging and minimum boat calculation problems.
"""

import logging
from typing import List, Tuple, Dict, Any

logger = logging.getLogger(__name__)


class SailingClubOptimizer:
    """
    Optimizes sailing club boat bookings by merging overlapping intervals
    and calculating minimum boats needed.
    """
    
    @staticmethod
    def merge_overlapping_intervals(intervals: List[List[int]]) -> List[List[int]]:
        """
        Merges overlapping booking intervals.
        
        Time Complexity: O(n log n) - dominated by sorting
        Space Complexity: O(n) - for result storage
        
        Args:
            intervals: List of [start, end] booking intervals
            
        Returns:
            List of merged non-overlapping intervals sorted by start time
        """
        if not intervals:
            return []
        
        # Sort intervals by start time
        sorted_intervals = sorted(intervals, key=lambda x: x[0])
        
        merged = [sorted_intervals[0]]
        
        for current in sorted_intervals[1:]:
            last_merged = merged[-1]
            
            # Check if current interval overlaps with the last merged interval
            if current[0] <= last_merged[1]:
                # Merge intervals by extending the end time
                merged[-1][1] = max(last_merged[1], current[1])
            else:
                # No overlap, add as new interval
                merged.append(current)
        
        return merged
    
    @staticmethod
    def calculate_minimum_boats(intervals: List[List[int]]) -> int:
        """
        Calculates minimum number of boats needed to satisfy all bookings.
        Uses sweep line algorithm with events.
        
        Time Complexity: O(n log n) - dominated by sorting events
        Space Complexity: O(n) - for events storage
        
        Args:
            intervals: List of [start, end] booking intervals
            
        Returns:
            Minimum number of boats needed
        """
        if not intervals:
            return 0
        
        # Create events: (time, event_type)
        # event_type: +1 for start, -1 for end
        events = []
        
        for start, end in intervals:
            events.append((start, 1))    # Booking starts (+1 boat needed)
            events.append((end, -1))     # Booking ends (-1 boat needed)
        
        # Sort events by time
        # For same time, process end events before start events
        # This ensures we free boats before needing new ones at the same time
        events.sort(key=lambda x: (x[0], x[1]))
        
        current_boats = 0
        max_boats_needed = 0
        
        for time, event_type in events:
            current_boats += event_type
            max_boats_needed = max(max_boats_needed, current_boats)
        
        return max_boats_needed
    
    @staticmethod
    def solve_test_case(test_case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solves a single test case from the sailing club challenge.
        
        Args:
            test_case: Dictionary containing 'id' and 'input' (list of intervals)
            
        Returns:
            Dictionary containing solution with 'id', 'sortedMergedSlots', and 'minBoatsNeeded'
        """
        try:
            test_id = test_case.get('id')
            intervals = test_case.get('input', [])
            
            if not test_id:
                raise ValueError("Test case missing required 'id' field")
            
            # Solve Part 1: Merge overlapping intervals
            merged_slots = SailingClubOptimizer.merge_overlapping_intervals(intervals)
            
            # Solve Part 2: Calculate minimum boats needed
            min_boats = SailingClubOptimizer.calculate_minimum_boats(intervals)
            
            return {
                'id': test_id,
                'sortedMergedSlots': merged_slots,
                'minBoatsNeeded': min_boats
            }
            
        except Exception as e:
            logger.error(f"Error solving test case {test_case.get('id', 'unknown')}: {e}")
            raise


def solve_sailing_club(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main function to process sailing club booking optimization request.
    
    Args:
        data: Request data containing 'testCases' list
        
    Returns:
        Response data containing 'solutions' list
    """
    try:
        test_cases = data.get('testCases', [])
        
        if not isinstance(test_cases, list):
            raise ValueError("Expected 'testCases' to be a list")
        
        solutions = []
        
        for test_case in test_cases:
            if not isinstance(test_case, dict):
                logger.warning(f"Skipping invalid test case: {test_case}")
                continue
                
            solution = SailingClubOptimizer.solve_test_case(test_case)
            solutions.append(solution)
        
        return {'solutions': solutions}
        
    except Exception as e:
        logger.error(f"Error in solve_sailing_club: {e}")
        raise