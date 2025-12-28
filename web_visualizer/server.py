"""
Flask server with WebSocket support for real-time Catan game visualization
"""
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from flask_cors import CORS

app = Flask(__name__)
app.config['SECRET_KEY'] = 'catan-visualizer-secret'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Store current game state
current_game_state = None

@app.route('/')
def index():
    """Serve the main visualization page"""
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    # Send current game state if available
    if current_game_state:
        emit('game_state', current_game_state)

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')

def broadcast_game_state(game_state):
    """Broadcast game state to all connected clients"""
    global current_game_state
    current_game_state = game_state
    socketio.emit('game_state', game_state)

def broadcast_game_end(result):
    """Broadcast game end result"""
    socketio.emit('game_end', result)

def run_server(host='0.0.0.0', port=5000):
    """Run the Flask server"""
    print(f"Starting visualization server at http://{host}:{port}")
    socketio.run(app, host=host, port=port, debug=False)

if __name__ == '__main__':
    run_server()
