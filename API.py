from Evaluation import *

# Create REST API
app = Flask(__name__)

# Define the available models
models = {
    'random_forest': rfc,
    'decision_tree': dtc,
    'neural_network': grid_result.best_estimator_,
}


# Define the API endpoint for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    input_data = request.json

    # Get the selected model from the request
    selected_model = input_data['model']

    # Get the input features from the request
    input_features = input_data['features']

    # Make a prediction using the selected model
    if selected_model not in models:
        return jsonify({'error': 'Invalid model selected'})
    else:
        prediction = models[selected_model](input_features)

    # Return the prediction as JSON
    return jsonify({'prediction': prediction})


app.run()
