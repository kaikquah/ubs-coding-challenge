"""
Test the sailing club endpoint locally
"""

import json
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import app

def test_sailing_club_endpoint():
    """Test the sailing club endpoint"""
    with app.test_client() as client:
        # Test data
        test_data = {
            "testCases": [
                {
                    "id": "test001",
                    "input": [[1, 8], [17, 28], [5, 8], [8, 10]]
                }
            ]
        }
        
        # Make POST request
        response = client.post(
            '/sailing-club/submission',
            data=json.dumps(test_data),
            content_type='application/json'
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        print(f"Response Data: {response.get_data(as_text=True)}")
        
        if response.status_code == 200:
            try:
                result = response.get_json()
                print(f"JSON Response: {json.dumps(result, indent=2)}")
                return True
            except Exception as e:
                print(f"Failed to parse JSON: {e}")
                return False
        else:
            print("Non-200 status code received")
            return False

if __name__ == "__main__":
    success = test_sailing_club_endpoint()
    if success:
        print("✅ Endpoint test passed!")
    else:
        print("❌ Endpoint test failed!")