#!/usr/bin/env python3
"""
Test script for the blankety challenge
"""

import requests
import json
import numpy as np

def create_test_data():
    """Create test data with known patterns"""
    test_series = []
    
    # Test case 1: Linear trend with noise
    x = np.linspace(0, 10, 1000)
    y = 2 * x + np.random.normal(0, 0.1, 1000)
    # Add some null values
    y_list = y.tolist()
    null_indices = np.random.choice(1000, 50, replace=False)
    for idx in null_indices:
        y_list[idx] = None
    test_series.append(y_list)
    
    # Test case 2: Sinusoidal pattern
    x = np.linspace(0, 4*np.pi, 1000)
    y = np.sin(x) + 0.5 * np.cos(2*x) + np.random.normal(0, 0.05, 1000)
    y_list = y.tolist()
    null_indices = np.random.choice(1000, 75, replace=False)
    for idx in null_indices:
        y_list[idx] = None
    test_series.append(y_list)
    
    # Add 98 more simple test cases
    for i in range(98):
        # Random polynomial + noise
        x = np.linspace(0, 1, 1000)
        coeffs = np.random.normal(0, 1, 3)
        y = coeffs[0] + coeffs[1]*x + coeffs[2]*x**2 + np.random.normal(0, 0.1, 1000)
        y_list = y.tolist()
        null_indices = np.random.choice(1000, np.random.randint(20, 100), replace=False)
        for idx in null_indices:
            y_list[idx] = None
        test_series.append(y_list)
    
    return test_series

def test_local():
    """Test the endpoint locally"""
    url = "http://localhost:8080/blankety"
    
    print("Creating test data...")
    test_data = create_test_data()
    
    payload = {"series": test_data}
    
    print("Sending request to local server...")
    try:
        response = requests.post(url, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if 'answer' in result:
                print("✅ Success! Received complete imputed series.")
                
                # Basic validation
                answer = result['answer']
                if len(answer) == 100:
                    print(f"✅ Correct number of series: {len(answer)}")
                else:
                    print(f"❌ Wrong number of series: {len(answer)}, expected 100")
                
                # Check first series
                if len(answer[0]) == 1000:
                    print(f"✅ First series has correct length: {len(answer[0])}")
                    
                    # Check for null values
                    has_nulls = any(x is None for x in answer[0])
                    if not has_nulls:
                        print("✅ No null values in imputed series")
                    else:
                        print("❌ Still has null values")
                else:
                    print(f"❌ First series wrong length: {len(answer[0])}, expected 1000")
                    
            else:
                print(f"❌ No 'answer' field in response: {result}")
        else:
            print(f"❌ Request failed with status {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server. Make sure the Flask app is running on localhost:8080")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_local()