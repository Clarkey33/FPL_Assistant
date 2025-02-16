'''Creating the flask app for my Fpl Assistant; first attempt'''

from flask import Flask, jsonify, request
import xgboost as xgb
import pandas as pd

# Initialize an instance of Flask
app = Flask(__name__)

# Load the trained model
model = xgb.XGBRegressor()
model.load_model('xgboost_fpl_model.json') 

@app.route('/')
def home():
    return "Welcome to the FPL Assistant!"

# Define a prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the request
        data = request.get_json(force=True)
        
        # Convert the data into a DataFrame
        input_data = pd.DataFrame([data])
        
        # Make predictions
        predictions = model.predict(input_data)
        
        # Return the predictions as JSON
        return jsonify({'predictions': predictions.tolist()})
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)