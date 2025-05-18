# Phishing URL Detector with AI Mode

A web-based tool that uses machine learning to detect phishing websites in real-time. The application analyzes URL features and provides instant feedback on potential threats.

## Features

- User-friendly interface for URL scanning
- AI-powered detection using supervised machine learning
- Real-time threat analysis with confidence scores
- Visual AI Mode with color-coded risk indicators
- RESTful API for model inference

## Tech Stack

- Frontend: HTML5, CSS3, JavaScript
- Backend: Python, Flask
- ML Libraries: scikit-learn, pandas, numpy
- Dataset: Phishing Website URLs Dataset

## Project Structure

```
├── frontend/           # Web interface files
│   ├── css/           # Stylesheets
│   ├── js/            # JavaScript files
│   └── index.html     # Main HTML file
├── backend/           # Flask server and ML components
│   ├── app.py         # Flask application
│   ├── model/         # ML model and training scripts
│   └── utils/         # Helper functions
├── data/              # Dataset and preprocessing
└── requirements.txt   # Python dependencies
```

## Setup Instructions

1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Start the Flask server:
   ```bash
   python backend/app.py
   ```

3. Open the web interface in your browser at `http://localhost:5000`

## Model Details

- Feature extraction from URLs (length, special characters, domain info)
- Classification using Random Forest/Decision Tree/Logistic Regression
- Performance metrics: Accuracy, Precision, Recall, F1-score

## API Endpoints

- `POST /api/predict`: Analyze URL for phishing threats
  ```json
  {
    "url": "https://example.com"
  }
  ```

## License

MIT License