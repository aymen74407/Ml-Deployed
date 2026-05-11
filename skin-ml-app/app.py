from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
import json
import logging
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Base directory for absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', 'Skin_Type_dataset.csv')

def get_dataset_stats():
    if not os.path.exists(DATA_PATH):
        return {}
    df = pd.read_csv(DATA_PATH)
    stats = {
        'total_rows': len(df),
        'total_features': len(df.columns),
        'skin_type_dist': df['Skin_Type'].value_counts().to_dict(),
        'gender_dist': df['Gender'].value_counts().to_dict(),
        'avg_age': round(float(df['Age'].mean()), 1),
        'avg_skin_score': round(float(df['skin_score'].mean()), 1),
        'avg_skin_age': round(float(df['skin_age'].mean()), 1),
    }
    return stats

@app.route('/')
def index():
    stats = get_dataset_stats()
    return render_template('index.html', stats=stats)

@app.route('/classification')
def classification():
    return render_template('classification.html')

@app.route('/clustering')
def clustering():
    return render_template('clustering.html')

@app.route('/regression')
def regression():
    return render_template('regression.html')

@app.route('/recommendation')
def recommendation():
    return render_template('recommendation.html')

@app.route('/ann')
def ann():
    return render_template('ann.html')

@app.route('/cnn')
def cnn_page():
    return render_template('cnn.html')

# ──── API: Classification ────
@app.route('/api/classification/predict', methods=['POST'])
def api_classification_predict():
    from models.classification import predict_single, encode_input
    try:
        data = request.get_json()
        algo = data.get('algorithm', 'random_forest')
        params = data.get('params', {})
        user_input = data.get('input', {})
        encoded = encode_input(user_input)
        result = predict_single(algo, encoded, params)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ──── API: Clustering ────
@app.route('/api/clustering/predict', methods=['POST'])
def api_clustering_predict():
    from models.clustering import predict_cluster, encode_input
    try:
        data = request.get_json()
        algo = data.get('algorithm', 'kmeans')
        params = data.get('params', {})
        user_input = data.get('input', {})
        encoded = encode_input(user_input)
        result = predict_cluster(algo, encoded, params)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ──── API: Regression ────
@app.route('/api/regression/predict', methods=['POST'])
def api_regression_predict():
    from models.regression import predict_single_regression, encode_input
    try:
        data = request.get_json()
        algo = data.get('algorithm', 'random_forest')
        target = data.get('target', data.get('target_col', 'skin_score'))
        params = data.get('params', {})
        user_input = data.get('input', {})
        encoded = encode_input(user_input)
        result = predict_single_regression(algo, target, encoded, params)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ──── API: Recommendation ────
@app.route('/api/recommendation/get', methods=['POST'])
def api_recommendation_get():
    from models.recommendation import get_recommendations
    try:
        data = request.get_json()
        user_input = data.get('input', {})
        result = get_recommendations(user_input)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ──── API: ANN ────
@app.route('/api/ann/predict', methods=['POST'])
def api_ann_predict():
    from models.ann import predict_ann, encode_input
    try:
        data = request.get_json()
        params = data.get('params', {})
        user_input = data.get('input', {})
        encoded = encode_input(user_input)
        result = predict_ann(encoded, params)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ──── API: CNN (image upload) ────
@app.route('/api/cnn/predict', methods=['POST'])
def api_cnn_predict():
    try:
        from models.cnn import predict_cnn
    except ImportError as e:
        logger.error("[CNN] ImportError — TensorFlow not available:\n%s", traceback.format_exc())
        return jsonify({
            'error': (
                'CNN requires TensorFlow, which is not installed in this deployment. '
                f'Details: {e}'
            )
        }), 503

    if 'image' not in request.files:
        return jsonify({'error': 'No image file in request (expected field name: "image")'}), 400

    try:
        file_bytes = request.files['image'].read()
        if len(file_bytes) == 0:
            return jsonify({'error': 'Uploaded file is empty'}), 400
        params = {
            'h1': request.form.get('h1', 128),
            'h2': request.form.get('h2', 64),
        }
        result = predict_cnn(file_bytes, params)
        return jsonify(result)
    except FileNotFoundError as e:
        logger.error("[CNN] Model file not found: %s", e)
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        logger.error("[CNN] Prediction failed:\n%s", traceback.format_exc())
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
