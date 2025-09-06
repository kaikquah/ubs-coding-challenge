import json
import math
sampleInputString = '''{
  "customers": [
    {
      "name": "CUSTOMER_A",
      "vip_status": false,
      "location": [
        1,
        1
      ],
      "credit_card": "CREDIT_CARD_1"
    },
    {
      "name": "CUSTOMER_B",
      "vip_status": false,
      "location": [
        2,
        -3
      ],
      "credit_card": "CREDIT_CARD_2"
    }
  ],
  "concerts": [
    {
      "name": "CONCERT_1",
      "booking_center_location": [
        1,
        5
      ]
    },
    {
      "name": "CONCERT_2",
      "booking_center_location": [
        -5,
        -3
      ]
    }
  ],
  "priority": {
    "CREDIT_CARD_1": "CONCERT_1",
    "CREDIT_CARD_2": "CONCERT_2"
  }
}'''
sampleInput = json.loads(sampleInputString)


def calculate_distance(customer_coord, concert_coord):
    """
    Calculate Euclidean distance between customer and concert coordinates
    
    Args:
        customer_coord: tuple (x, y) - customer coordinates
        concert_coord: tuple (x, y) - concert coordinates
    
    Returns:
        float: Distance between the two points
    """
    x1, y1 = customer_coord
    x2, y2 = concert_coord
    
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

def distance_to_points_linear(distance, max_distance=6.0):
    if distance >= max_distance:
        return 0
    points = 30 * (1 - distance / max_distance)
    return max(0, int(round(points)))

def processInput(data):
    customers = data["customers"]
    concerts = data["concerts"]
    creditCardPrio = data["priority"]

    highestPrioTable = {}

    for customer in customers:
        custName = customer['name']
        custCoord = customer['location']
        isVip = customer['vip_status']
        largestPoints = -10
        mostLikely = None
        
        for concert in concerts:
            concertPoints = 0
            
            # Check concert distance
            dist = calculate_distance(custCoord, concert['booking_center_location'])
            distBonus = distance_to_points_linear(dist)
            
            # Check credit card priority - FIXED
            creditPoints = 0
            if concert['name'] == creditCardPrio.get(customer['credit_card']):
                creditPoints += 50
            
            # Check VIP status
            vipPoints = 0
            if isVip:
                vipPoints += 100
                
            concertPoints = distBonus + creditPoints + vipPoints
            
            if concertPoints >= largestPoints:
                largestPoints = max(largestPoints, concertPoints)
                mostLikely = concert['name']
                
        highestPrioTable[custName] = mostLikely
    
    return highestPrioTable    
if __name__ == "__main__":
    processInput(sampleInput)

            
