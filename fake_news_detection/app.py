"""
Fake News Detection using NLP
A Flask-based web application for detecting fake news using machine learning
Author: Expert Developer
Purpose: Class Project
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pandas as pd
import os
from datetime import datetime
import json
import warnings

warnings.filterwarnings('ignore')

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variables for models
models = {}
vectorizer = None
model_stats = {
    'total_predictions': 0,
    'fake_detected': 0,
    'real_detected': 0,
    'accuracy_scores': []
}

class FakeNewsDetector:
    """
    A comprehensive class for detecting fake news using multiple ML algorithms
    """
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.models = {
            'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'naive_bayes': MultinomialNB(),
            'svm': SVC(kernel='linear', probability=True, random_state=42)
        }
        self.trained = False
        self.model_performance = {}
        
    def train(self, texts, labels):
        """
        Train all models with provided data
        
        Args:
            texts: List of news articles
            labels: List of labels (0 for fake, 1 for real)
        """
        try:
            print("Starting training process...")
            
            # Vectorize the text data
            X = self.vectorizer.fit_transform(texts)
            y = np.array(labels)
            
            # Train each model
            for model_name, model in self.models.items():
                print(f"Training {model_name}...")
                model.fit(X, y)
                
                # Calculate training accuracy
                accuracy = model.score(X, y)
                self.model_performance[model_name] = accuracy
                print(f"{model_name} - Training Accuracy: {accuracy:.4f}")
            
            self.trained = True
            print("Training completed successfully!")
            return True
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            return False
    
    def predict(self, text, model_name='logistic_regression'):
        """
        Predict if a news article is fake or real
        
        Args:
            text: News article text
            model_name: Name of the model to use
            
        Returns:
            Dictionary with prediction results
        """
        try:
            if not self.trained:
                return {'error': 'Model not trained yet'}
            
            # Vectorize input text
            X = self.vectorizer.transform([text])
            
            # Get prediction from selected model
            model = self.models[model_name]
            prediction = model.predict(X)[0]
            
            # Get probability scores
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X)[0]
                confidence = max(probabilities) * 100
            else:
                confidence = 0
            
            result = {
                'prediction': 'REAL NEWS ✓' if prediction == 1 else 'FAKE NEWS ✗',
                'class': prediction,
                'confidence': round(confidence, 2),
                'model_used': model_name,
                'status': 'success'
            }
            
            return result
            
        except Exception as e:
            return {'error': str(e), 'status': 'error'}
    
    def predict_ensemble(self, text):
        """
        Use ensemble method combining all models
        
        Args:
            text: News article text
            
        Returns:
            Dictionary with ensemble prediction
        """
        try:
            X = self.vectorizer.transform([text])
            predictions = []
            confidences = []
            
            for model_name, model in self.models.items():
                pred = model.predict(X)[0]
                predictions.append(pred)
                
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(X)[0]
                    confidences.append(max(prob))
            
            # Majority voting
            ensemble_prediction = 1 if sum(predictions) > len(predictions) / 2 else 0
            average_confidence = (sum(confidences) / len(confidences)) * 100
            
            result = {
                'prediction': 'REAL NEWS ✓' if ensemble_prediction == 1 else 'FAKE NEWS ✗',
                'class': ensemble_prediction,
                'confidence': round(average_confidence, 2),
                'model_used': 'Ensemble (Voting)',
                'individual_predictions': {
                    model_name: 'Real' if pred == 1 else 'Fake'
                    for model_name, pred in zip(self.models.keys(), predictions)
                },
                'status': 'success'
            }
            
            return result
            
        except Exception as e:
            return {'error': str(e), 'status': 'error'}
    
    def get_model_stats(self):
        """Get statistics about trained models"""
        return {
            'trained': self.trained,
            'models': list(self.models.keys()),
            'performance': self.model_performance,
            'vectorizer_features': self.vectorizer.get_feature_names_out().tolist() if self.trained else []
        }


# Initialize detector
detector = FakeNewsDetector()

def load_training_data():
    """
    Load and prepare training data
    Uses sample data for demonstration
    """
    # Sample training data (in production, use larger datasets like LIAR or FakeNewsNet)
    sample_data = [
        ("Breaking: Secret cure for cancer discovered but hidden by big pharma! Click now!", 0),
        ("Scientists announce breakthrough in renewable energy research", 1),
        ("SHOCKING: All water is being replaced with chemicals! URGENT!", 0),
        ("New study shows benefits of regular exercise and healthy diet", 1),
        ("Politicians caught in secret meeting! What they discussed will shock you!", 0),
        ("WHO releases new guidelines for public health", 1),
        ("EXCLUSIVE: Area 51 alien photos leaked by NASA insider", 0),
        ("Investment firm reports quarterly earnings of 2.5 billion dollars", 1),
        ("Miracle weight loss pill discovered! Lose 30 pounds in a week!", 0),
        ("University research team develops new software algorithm", 1),
        ("5G towers causing COVID-19 pandemic confirmed by doctors", 0),
        ("New smartphone model features improved camera technology", 1),
        ("Government hiding proof of time travel for 50 years!", 0),
        ("Stock market closes with moderate gains following inflation data", 1),
        ("This one weird trick eliminates belly fat overnight!", 0),
        ("Climate scientists publish report on temperature changes", 1),
        ("INCREDIBLE: Man grows third arm through meditation!", 0),
        ("Pharmaceutical company completes Phase 3 clinical trials", 1),
        ("PROOF: Reptilians control world governments!", 0),
        ("Technology conference announces new industry standards", 1),
    ]
    
    texts = [item[0] for item in sample_data]
    labels = [item[1] for item in sample_data]
    
    return texts, labels

# Initialize and train model at startup
@app.before_request
def setup():
    """Setup models before first request"""
    global detector
    if not detector.trained:
        texts, labels = load_training_data()
        detector.train(texts, labels)

# Routes
@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for single prediction"""
    try:
        data = request.json
        news_text = data.get('text', '').strip()
        model_type = data.get('model', 'ensemble')
        
        if not news_text:
            return jsonify({'error': 'Please enter news text', 'status': 'error'}), 400
        
        if len(news_text) < 10:
            return jsonify({'error': 'Text must be at least 10 characters long', 'status': 'error'}), 400
        
        # Get prediction
        if model_type == 'ensemble':
            result = detector.predict_ensemble(news_text)
        else:
            result = detector.predict(news_text, model_type)
        
        # Update statistics
        if result.get('status') == 'success':
            model_stats['total_predictions'] += 1
            if result['class'] == 0:
                model_stats['fake_detected'] += 1
            else:
                model_stats['real_detected'] += 1
            model_stats['accuracy_scores'].append(result['confidence'])
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """API endpoint for batch predictions"""
    try:
        data = request.json
        texts = data.get('texts', [])
        model_type = data.get('model', 'ensemble')
        
        if not texts or not isinstance(texts, list):
            return jsonify({'error': 'Please provide a list of texts', 'status': 'error'}), 400
        
        results = []
        for text in texts:
            if model_type == 'ensemble':
                result = detector.predict_ensemble(text)
            else:
                result = detector.predict(text, model_type)
            results.append(result)
        
        return jsonify({'predictions': results, 'status': 'success'})
        
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """Get application statistics"""
    stats = {
        'total_predictions': model_stats['total_predictions'],
        'fake_detected': model_stats['fake_detected'],
        'real_detected': model_stats['real_detected'],
        'average_confidence': round(
            sum(model_stats['accuracy_scores']) / len(model_stats['accuracy_scores']), 2
        ) if model_stats['accuracy_scores'] else 0,
        'models_available': detector.get_model_stats()['models'],
        'status': 'success'
    }
    return jsonify(stats)

@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """Get information about available models"""
    info = {
        'models': list(detector.models.keys()),
        'performance': detector.model_performance,
        'trained': detector.trained,
        'training_samples': 20,  # From our sample data
        'status': 'success'
    }
    return jsonify(info)

@app.route('/api/clear-stats', methods=['POST'])
def clear_statistics():
    """Clear all statistics"""
    global model_stats
    model_stats = {
        'total_predictions': 0,
        'fake_detected': 0,
        'real_detected': 0,
        'accuracy_scores': []
    }
    return jsonify({'message': 'Statistics cleared', 'status': 'success'})

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found', 'status': 'error'}), 404

@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error', 'status': 'error'}), 500

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║   FAKE NEWS DETECTION USING NLP - Flask Application          ║
    ║   Starting server...                                          ║
    ║   Access at: http://localhost:5000                            ║
    ║   APIs Available:                                             ║
    ║   - POST /api/predict - Single prediction                     ║
    ║   - POST /api/batch-predict - Multiple predictions            ║
    ║   - GET /api/statistics - Application statistics              ║
    ║   - GET /api/model-info - Model information                   ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    app.run(debug=True, host='localhost', port=5000)
