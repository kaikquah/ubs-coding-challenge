#!/usr/bin/env python3

from princess_diaries_segment_tree import solve_princess_diaries

# Test with the provided example
test_input = {
    "tasks": [
        {"name": "A", "start": 480, "end": 540, "station": 1, "score": 2},
        {"name": "B", "start": 600, "end": 660, "station": 2, "score": 1},
        {"name": "C", "start": 720, "end": 780, "station": 3, "score": 3},
        {"name": "D", "start": 840, "end": 900, "station": 4, "score": 1},
        {"name": "E", "start": 960, "end": 1020, "station": 1, "score": 4},
        {"name": "F", "start": 530, "end": 590, "station": 2, "score": 1}
    ],
    "subway": [
        {"connection": [0, 1], "fee": 10},
        {"connection": [1, 2], "fee": 10},
        {"connection": [2, 3], "fee": 20},
        {"connection": [3, 4], "fee": 30}
    ],
    "starting_station": 0
}

expected_output = {
    "max_score": 11,
    "min_fee": 140,
    "schedule": ["A", "B", "C", "D", "E"]
}

result = solve_princess_diaries(test_input)
print("Segment Tree Solution:")
print("Expected:", expected_output)
print("Actual  :", result)
print("Test passed:", result == expected_output)