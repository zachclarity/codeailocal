"""
Sample Flask auth middleware for demonstration.
This snippet gets auto-imported into RAG on startup.
"""
from flask import Flask, request, jsonify
from functools import wraps
import hashlib
import secrets

app = Flask(__name__)

# Simple token store
tokens = {}
users = {
    "admin": hashlib.sha256("password123".encode()).hexdigest()
}

def require_auth(f):
    """Decorator to require valid auth token."""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        if not token or token not in tokens:
            return jsonify({"error": "Unauthorized"}), 401
        request.user = tokens[token]
        return f(*args, **kwargs)
    return decorated

@app.route('/auth/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username', '')
    password = hashlib.sha256(data.get('password', '').encode()).hexdigest()
    
    if username in users and users[username] == password:
        token = secrets.token_hex(32)
        tokens[token] = username
        return jsonify({"token": token, "user": username})
    
    return jsonify({"error": "Invalid credentials"}), 401

@app.route('/auth/logout', methods=['POST'])
@require_auth
def logout():
    token = request.headers.get('Authorization', '').replace('Bearer ', '')
    tokens.pop(token, None)
    return jsonify({"message": "Logged out"})

@app.route('/protected', methods=['GET'])
@require_auth
def protected():
    return jsonify({"message": f"Hello {request.user}!", "data": "secret stuff"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
