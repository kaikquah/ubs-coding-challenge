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
import cv2
logger = logging.getLogger(__name__)
app = Flask(__name__)

# Register blueprints
app.register_blueprint(blankety_bp)
app.register_blueprint(latex_bp)
app.register_blueprint(snakes_bp) 

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
