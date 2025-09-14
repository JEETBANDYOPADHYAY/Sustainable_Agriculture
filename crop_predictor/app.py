import pandas as pd
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings

# Suppress all warnings to keep the console clean
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Load the pre-trained model and preprocessors
try:
    model = joblib.load('crop.pkl')
    le = joblib.load('le.pkl')
    scaler = joblib.load('scaler.pkl')
    carbon_data = joblib.load('carbon_data.pkl')
    
    crop_factors = carbon_data['crop_factor']
    low_threshold = carbon_data['low_threshold']
    high_threshold = carbon_data['high_threshold']
    crop_types = carbon_data['crop_types']

    print("Model and preprocessors loaded successfully!")

except FileNotFoundError as e:
    print(f"Error: {e}. Please ensure all .pkl files are in the directory.")
    model, le, scaler, crop_factors, low_threshold, high_threshold, crop_types = None, None, None, {}, 0, 0, []

def soil_factor(pH, moisture):
    """Calculates a simple soil factor based on pH and moisture."""
    return (pH + moisture) * 0.5

def categorize_carbon_emission(value):
    """Categorizes the carbon emission value into 'Low', 'Medium', or 'High'."""
    if value >= high_threshold:
        return 'High'
    elif value >= low_threshold:
        return 'Medium'
    else:
        return 'Low'

@app.route('/')
def home():
    """Renders the main HTML page."""
    return render_template('index.html')

@app.route('/api/croptypes')
def get_crop_types():
    """Returns the list of crop types as a JSON object for the frontend."""
    if not crop_types:
        return jsonify({'error': 'Crop types not loaded. Check server logs.'}), 500
    return jsonify({'crop_types': crop_types})

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the prediction request from the frontend."""
    if not model or not le or not scaler:
        return jsonify({'error': 'Model or preprocessors not loaded. Check server logs.'}), 500

    data = request.get_json()
    try:
        soil_ph = float(data['Soil_pH'])
        soil_moisture = float(data['Soil_Moisture'])
        temperature_c = float(data['Temperature_C'])
        rainfall_mm = float(data['Rainfall_mm'])
        crop_type = data['Crop_Type']
        fertilizer_usage = float(data['Fertilizer_Usage_kg'])
        pesticide_usage = float(data['Pesticide_Usage_kg'])
        sustainability_score = float(data['Sustainability_Score'])
    except (KeyError, ValueError) as e:
        return jsonify({'error': f"Invalid input data: {e}"}), 400

    # Create a DataFrame for prediction, matching the feature order used in training
    # Sustainability_Score and Crop_Type_Encoded are included in the correct order
    input_df = pd.DataFrame([[
        soil_ph, soil_moisture, temperature_c, rainfall_mm, fertilizer_usage, 
        pesticide_usage, sustainability_score, le.transform([crop_type])[0]
    ]], columns=['Soil_pH', 'Soil_Moisture', 'Temperature_C', 'Rainfall_mm', 
                 'Fertilizer_Usage_kg', 'Pesticide_Usage_kg', 'Sustainability_Score', 'Crop_Type_Encoded'])

    # Scale the input data using the pre-trained scaler
    input_scaled = scaler.transform(input_df)

    # Make the prediction
    predicted_yield = model.predict(input_scaled)[0]

    # Carbon Emission Calculation
    carbon_emission = (
        fertilizer_usage * 1.2 +
        pesticide_usage * 0.8 +
        rainfall_mm * crop_factors.get(crop_type, 0.01) +
        soil_factor(soil_ph, soil_moisture)
    )
    
    carbon_category = categorize_carbon_emission(carbon_emission)

    return jsonify({
        'crop_yield': f'{predicted_yield:.2f}',
        'carbon_emission': f'{carbon_emission:.2f}',
        'emission_category': carbon_category
    })

if __name__ == '__main__':
    # Disable the reloader to prevent the SystemExit error on some systems
    app.run(debug=True, port=8000, use_reloader=False)
