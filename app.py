import logging
import socket
import os
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

# if __name__ == "__main__":
#     logging.info("Starting application ...")
#     sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     sock.bind(('localhost', 8080))
#     port = sock.getsockname()[1]
#     sock.close()
#     app.run(port=port)

if __name__ == "__main__":
    logging.info("Starting application ...")
    
    # Get port from environment variable (Render provides this)
    # Fallback to 8080 for local development
    port = int(os.environ.get("PORT", 8080))
    
    # For Render deployment, bind to 0.0.0.0 (not localhost)
    # For local development, you can use localhost
    host = "0.0.0.0" if os.environ.get("RENDER") else "localhost"
    
    app.run(host=host, port=port, debug=False)