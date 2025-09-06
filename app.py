import logging
import socket
import os
from flask import Flask, jsonify, request
from TicketingAgent import calculate_distance, distance_to_points_linear, processInput
from blankety_challenge import blankety_bp
from mst_solver import calculate_mst_weights
from princess_diaries_optimized import solve_princess_diaries

logger = logging.getLogger(__name__)
app = Flask(__name__)

# Register blueprints
app.register_blueprint(blankety_bp)

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
