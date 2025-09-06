import logging
import socket
import os
from flask import Flask, jsonify, request
from TicketingAgent import calculate_distance, distance_to_points_linear, processInput
from blankety_challenge import blankety_bp
from spy import buildAdjacencyList, process_data, processNetwork
from mst_solver import calculate_mst_weights
from princess_diaries_optimized import solve_princess_diaries
from latex_formula_evaluator import latex_bp
from snakes_ladders_solver import snakes_bp
from sailing_club import sailing_bp
import cv2
from ink_archive_solver import solve_ink_archive_challenge
from mages_gambit_solver import solve_mages_gambit_multiple

from sailing_club import merge_bookings, min_boats_needed
from duolingo_sort import solve_duolingo_sort
from sailing_club import merge_bookings, min_boats_needed
from duolingo_sort import solve_duolingo_sort
from fog_of_wall_solver import fog_bp

logger = logging.getLogger(__name__)
app = Flask(__name__)

# Register blueprints
app.register_blueprint(blankety_bp)
app.register_blueprint(latex_bp)
app.register_blueprint(snakes_bp) 
app.register_blueprint(sailing_bp)
app.register_blueprint(fog_bp)

@app.before_request
def log_request_info():
    logger.info('Request: %s %s', request.method, request.url)
    logger.info('Headers: %s', dict(request.headers))
    if request.is_json:
        logger.info('Request Body: %s', request.get_json())
    else:
        logger.info('Request Data: %s', request.get_data(as_text=True))
@app.after_request
def log_response_info(response):
    logger.info('Response Status: %s', response.status)
    logger.info('Response Headers: %s', dict(response.headers))
    return response

@app.route('/', methods=['GET'])
def default_route():
    return 'Python Template'

@app.route("/trivia", methods = ["GET"])
def trivia():
    answers = [
        4,
        1,
        2,
        2,
        3,
        4,
        4,
        5,
        4,
        3,
        3,
        2,
        1,
        4,
        2,
        1,
        1,
        2,
        2,
        1,
        5,
        2,
        3,
        3,
        2
    ]
    return jsonify({"answers":answers})

        
    
@app.route('/debug-routes', methods=['GET'])
def debug_routes():
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append({
            'endpoint': rule.endpoint,
            'methods': list(rule.methods),
            'rule': str(rule)
        })
    return jsonify(routes)


logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


# Ticketing agent api
@app.route('/ticketing-agent', methods = ['POST'])
def ticket_agent():
    if request.content_type != 'application/json':
        return jsonify({
            'error': 'Content-Type must be application/json'
        }), 400
    data = request.get_json()
    result = processInput(data)
    response = jsonify(result)
    return response

# spy network
# spy network
@app.route('/investigate', methods=['POST'])
def investigate():
    try:
        logger.info("=== INVESTIGATE ENDPOINT CALLED ===")
        
        # Check content type
        if request.content_type != 'application/json':
            logger.error(f"Invalid content type: {request.content_type}")
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        
        # Parse JSON
        data = request.get_json()
        logger.info(f"Received data: {data}")
        
        if not data:
            logger.error("No JSON data received")
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Handle the actual data structure: direct list of networks
        if isinstance(data, list):
            # Challenge sends: [{"networkId": "...", "network": [...]}, ...]
            networks_data = {"networks": data}
            logger.info(f"Converted list to networks structure: {len(data)} networks")
        elif isinstance(data, dict) and 'networks' in data:
            # Standard structure: {"networks": [...]}
            networks_data = data
            logger.info("Using standard networks structure")
        else:
            logger.error(f"Unexpected data structure: {type(data)}, keys: {list(data.keys()) if isinstance(data, dict) else 'not a dict'}")
            return jsonify({'error': 'Invalid data structure'}), 400
        
        # Process the data
        logger.info("Processing spy network data...")
        result = process_data(networks_data)
        logger.info(f"Result: {result}")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"EXCEPTION in /investigate: {str(e)}")
        import traceback
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500
    



@app.route('/the-mages-gambit', methods=['POST'])
def mages_gambit():
    logger.info("Mage's Gambit endpoint called")
    
    if request.content_type != 'application/json':
        logger.error(f"Invalid content type: {request.content_type}")
        return jsonify({'error': 'Content-Type must be application/json'}), 400
    
    try:
        data = request.get_json()
        logger.info(f"Received Mage's Gambit data: {data}")
        
        if not isinstance(data, list):
            logger.error("Expected a list of test cases")
            return jsonify({'error': 'Expected a list of test cases'}), 400
        
        # Validate structure
        for i, test_case in enumerate(data):
            if not isinstance(test_case, dict):
                logger.error(f"Test case {i} is not a dict")
                return jsonify({'error': f'Test case {i} must be a dictionary'}), 400
            
            required_fields = ['intel', 'reserve', 'fronts', 'stamina']
            for field in required_fields:
                if field not in test_case:
                    logger.error(f"Test case {i} missing field: {field}")
                    return jsonify({'error': f'Test case {i} must have "{field}" field'}), 400
        
        # Process the test cases
        logger.info("Processing Mage's Gambit challenges...")
        result = solve_mages_gambit_multiple(data)
        logger.info(f"Mage's Gambit result: {result}")
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error in /the-mages-gambit: {e}")
        import traceback
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        return jsonify({'error': 'Internal server error', 'message': str(e)}), 500

@app.route('/The-Ink-Archive', methods=['POST'])
def ink_archive():
    logger.info("Ink Archive endpoint called")
    
    if request.content_type != 'application/json':
        logger.error(f"Invalid content type: {request.content_type}")
        return jsonify({'error': 'Content-Type must be application/json'}), 400
    
    try:
        data = request.get_json()
        logger.info(f"Received Ink Archive data with {len(data)} challenges")
        
        if not isinstance(data, list):
            logger.error("Expected a list of challenges")
            return jsonify({'error': 'Expected a list of challenges'}), 400
        
        # Validate structure
        for i, challenge in enumerate(data):
            if not isinstance(challenge, dict):
                logger.error(f"Challenge {i} is not a dict")
                return jsonify({'error': f'Challenge {i} must be a dictionary'}), 400
            
            if 'goods' not in challenge or 'ratios' not in challenge:
                logger.error(f"Challenge {i} missing required fields")
                return jsonify({'error': f'Challenge {i} must have "goods" and "ratios" fields'}), 400
        
        # Process the challenges using the improved algorithm
        logger.info("Processing Ink Archive challenges with improved algorithms...")
        result = solve_ink_archive_challenge(data)
        logger.info(f"Ink Archive result: {result}")
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error in /The-Ink-Archive: {e}")
        import traceback
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        return jsonify({'error': 'Internal server error', 'message': str(e)}), 500
# MST calculation endpoint
@app.route('/mst-calculation', methods=['POST'])
def mst_calculation():
    if request.content_type != 'application/json':
        return jsonify({
            'error': 'Content-Type must be application/json'
        }), 400
    
    try:
        data = request.get_json()
        if not isinstance(data, list):
            return jsonify({
                'error': 'Expected a list of test cases'
            }), 400
        
        # Process the test cases and calculate MST weights
        results = calculate_mst_weights(data)
        
        return jsonify(results)
    
    except Exception as e:
        logger.error(f"Error in MST calculation: {e}")
        return jsonify({
            'error': 'Internal server error'
        }), 500


# Princess Diaries endpoint
@app.route('/princess-diaries', methods=['POST'])
def princess_diaries():
    logger.info("Princess Diaries endpoint called")
    if request.content_type != 'application/json':
        logger.error(f"Invalid content type: {request.content_type}")
        return jsonify({'error': 'Content-Type must be application/json'}), 400
    try:
        data = request.get_json()
        logger.info(f"Received data: {data}")
        if not isinstance(data, dict):
            return jsonify({'error': 'Invalid payload format'}), 400
        result = solve_princess_diaries(data)
        logger.info(f"Returning result: {result}")
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in /princess-diaries: {e}")
        return jsonify({'error': 'Internal server error'}), 500

# Duolingo Sort endpoint
@app.route('/duolingo-sort', methods=['POST'])
def duolingo_sort():
    logger.info("Duolingo Sort endpoint called")
    
    # Check if content type contains 'application/json' (allows charset)
    if not request.is_json:
        logger.error(f"Invalid content type: {request.content_type}")
        return jsonify({'error': 'Content-Type must be application/json'}), 400
    
    try:
        data = request.get_json()
        logger.info(f"Received Duolingo Sort data: Part {data.get('part')}, Challenge {data.get('challenge')}")
        
        # Validate input structure
        if 'part' not in data or 'challengeInput' not in data:
            logger.error("Missing required fields")
            return jsonify({'error': 'Missing required fields: part or challengeInput'}), 400
        
        if 'unsortedList' not in data['challengeInput']:
            logger.error("Missing unsortedList in challengeInput")
            return jsonify({'error': 'Missing unsortedList in challengeInput'}), 400
        
        # Process the sorting challenge
        logger.info(f"Processing {len(data['challengeInput']['unsortedList'])} items")
        result = solve_duolingo_sort(data)
        logger.info(f"Duolingo Sort result: {result}")
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error in /duolingo-sort: {e}")
        import traceback
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        return jsonify({'error': 'Internal server error', 'message': str(e)}), 500

# Add a simple test route to verify the app is working

@app.route('/test', methods=['GET', 'POST'])
def test_route():
    return jsonify({
        'message': 'App is working',
        'method': request.method,
        'routes_count': len(list(app.url_map.iter_rules()))
    })

if __name__ == "__main__":
    logging.info("Starting application in development mode...")
    
    # Log all registered routes for debugging
    logger.info("Registered routes:")
    for rule in app.url_map.iter_rules():
        logger.info(f"  {rule.endpoint}: {rule.rule} {list(rule.methods)}")
    
    # This only runs when you execute: python app.py
    # Render will use Gunicorn instead
    port = int(os.environ.get("PORT", 8080))
    app.run(host="localhost", port=port, debug=True)
