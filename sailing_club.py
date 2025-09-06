from flask import Blueprint, jsonify, request

sailing_bp = Blueprint("sailing_club", __name__)

def merge_bookings(bookings):
    # Sort bookings by start time
    bookings.sort(key=lambda x: x[0])
    merged_slots = []
    
    for booking in bookings:
        if not merged_slots or merged_slots[-1][1] < booking[0]:
            merged_slots.append(booking)
        else:
            merged_slots[-1][1] = max(merged_slots[-1][1], booking[1])
    
    return merged_slots

def min_boats_needed(bookings):
    events = []
    
    # Collect all start and end times as events
    for booking in bookings:
        events.append((booking[0], 'start'))
        events.append((booking[1], 'end'))
    
    # Sort events, prioritizing 'start' over 'end' in case of tie
    events.sort(key=lambda x: (x[0], x[1] == 'end'))
    
    boats_needed = 0
    max_boats = 0
    
    for event in events:
        if event[1] == 'start':
            boats_needed += 1
            max_boats = max(max_boats, boats_needed)
        else:
            boats_needed -= 1
    
    return max_boats

@sailing_bp.route('/submission', methods=['POST'])
def sailing_submission():
    data = request.get_json()
    
    solutions = []
    
    for test_case in data['testCases']:
        bookings = test_case['input']
        
        # Part 1: Merge overlapping bookings
        sorted_merged_slots = merge_bookings(bookings)
        
        # Part 2: Calculate minimum boats needed
        min_boats = min_boats_needed(bookings)
        
        solutions.append({
            'id': test_case['id'],
            'sortedMergedSlots': sorted_merged_slots,
            'minBoatsNeeded': min_boats
        })
    
    return jsonify({'solutions': solutions})
