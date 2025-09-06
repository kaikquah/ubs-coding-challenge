import logging
import socket
import os
from flask import Flask, jsonify, request
from TicketingAgent import calculate_distance, distance_to_points_linear, processInput
from flask import Flask
from blankety_challenge import blankety_bp

logger = logging.getLogger(__name__)
app = Flask(__name__)

# Register blueprints
app.register_blueprint(blankety_bp)

@app.route('/', methods=['GET'])
def default_route():
    return 'Python Template'


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
if __name__ == "__main__":
    logging.info("Starting application in development mode...")
    
    # This only runs when you execute: python app.py
    # Render will use Gunicorn instead
    port = int(os.environ.get("PORT", 8080))
    app.run(host="localhost", port=port, debug=True)