from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
from utils.feature_extractor import URLFeatureExtractor

app = Flask(__name__)
CORS(app)

# Initialize feature extractor
feature_extractor = URLFeatureExtractor()

# Load the ML model (will be implemented later)
model = None

@app.route('/api/analyze', methods=['POST'])
def analyze_url():
    try:
        data = request.get_json()
        url = data.get('url')
        
        if not url:
            return jsonify({'error': 'URL is required'}), 400
            
        # Extract features from URL
        features = feature_extractor.extract_features(url)
        
        # For now, return dummy response until model is implemented
        response = {
            'url': url,
            'is_phishing': False,  # Will be replaced with actual prediction
            'confidence': 0.85,    # Will be replaced with actual confidence
            'features': {
                'url_length': len(url),
                'has_https': url.startswith('https'),
                'has_at_symbol': '@' in url,
                # More features will be added
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    return jsonify({'status': 'online'})

if __name__ == '__main__':
    app.run(debug=True)