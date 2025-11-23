from flask import Flask, render_template, request, send_file
import pandas as pd
from datetime import datetime
import os
import sys
import csv

app = Flask(__name__)

# Load the model with compatibility handling
model = None
model_path = os.path.join(os.path.dirname(__file__), 'Iced_Coffee_Demand_Prediction.joblib')

print(f"\n=== MODEL LOADING ===")
print(f"Looking for model at: {model_path}")
print(f"File exists: {os.path.exists(model_path)}")

if os.path.exists(model_path):
    print(f"File size: {os.path.getsize(model_path)} bytes")
    try:
        import joblib
        print("Attempting to load with joblib (unsafe_load=True)...")
        model = joblib.load(model_path, unsafe_load=True)
        print("✓ Model loaded successfully with unsafe_load=True!")
    except TypeError as te:
        print(f"TypeError (trying without unsafe_load): {te}")
        try:
            import joblib
            print("Attempting to load with joblib (no unsafe_load)...")
            model = joblib.load(model_path)
            print("✓ Model loaded successfully!")
        except Exception as e:
            print(f"✗ Joblib failed: {e}")
            import traceback
            traceback.print_exc()
    except Exception as e:
        print(f"Error with unsafe_load: {e}")
        import traceback
        traceback.print_exc()
        try:
            import pickle
            print("Attempting to load with pickle...")
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print("✓ Model loaded successfully with pickle!")
        except Exception as e2:
            print(f"✗ Pickle also failed: {e2}")
            traceback.print_exc()
else:
    print(f"✗ File not found at: {model_path}")
    print(f"Looking in: {os.path.dirname(__file__)}")
    print("Files in directory:")
    for f in os.listdir(os.path.dirname(__file__) or '.'):
        print(f"  - {f}")

print("=== END ===\n")

# CSV file to store predictions
CSV_FILE = 'predictions.csv'
CSV_HEADERS = ['Timestamp', 'Month', 'Day', 'Year', 'Day_of_Week', 'Time_of_Day', 'Temperature', 
               'Weather_Condition', 'Holiday', 'University_Event', 'Predicted_Cups_Sold']

# Create CSV file with headers if it doesn't exist
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADERS)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    error = None
    
    if request.method == 'POST':
        try:
            if model is None:
                error = "Model not loaded. Please ensure 'Iced_Coffee_Demand_Prediction.joblib' exists."
            else:
                # Get form data
                month = request.form.get('month', '').strip()
                day = request.form.get('day', '').strip()
                year = request.form.get('year', '').strip()
                time_of_day = request.form.get('time_of_day', '').strip()
                temperature = request.form.get('temperature', '').strip()
                weather = request.form.get('weather', '').strip()
                holiday = request.form.get('holiday', '').strip()
                university_event = request.form.get('university_event', '').strip()
                
                print(f"POST received: month={repr(month)}, day={repr(day)}, year={repr(year)}, temp={repr(temperature)}")
                
                # Validate required fields
                if not month or not day or not year or not temperature or not time_of_day:
                    error = "Please fill all required fields"
                else:
                    try:
                        month_int = int(month)
                        day_int = int(day)
                        year_int = int(year)
                        temp_float = float(temperature)
                        
                        # Validate ranges
                        if not (1 <= month_int <= 12):
                            error = "Month must be between 1 and 12"
                        elif not (1 <= day_int <= 31):
                            error = "Day must be between 1 and 31"
                        elif year_int < 2000 or year_int > 2100:
                            error = "Year must be between 2000 and 2100"
                        elif temp_float < -50 or temp_float > 60:
                            error = "Temperature must be between -50 and 60°C"
                        else:
                            # Map month number to month name
                            month_names = ['', 'January', 'February', 'March', 'April', 'May', 'June',
                                         'July', 'August', 'September', 'October', 'November', 'December']
                            month_name = month_names[month_int]
                            
                            # Calculate day of week name
                            date_obj = datetime(year_int, month_int, day_int)
                            days_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                            day_of_week_name = days_names[date_obj.weekday()]
                            
                            # Map time of day to match training data format
                            time_mapping = {'morning': 'Morning', 'afternoon': 'Afternoon', 'evening': 'Evening'}
                            time_of_day_name = time_mapping.get(time_of_day.lower(), 'Morning')
                            
                            # Keep weather and holiday as-is or use defaults
                            weather_condition = weather if weather else 'Sunny'
                            holiday_status = holiday if holiday else 'Regular Day'
                            university_event_status = university_event if university_event else 'Regular Day'
                            
                            # Prepare input data - match model's expected feature names and format
                            input_data = pd.DataFrame({
                                'Month': [month_name],
                                'Day': [day_int],
                                'Day_of_Week': [day_of_week_name],
                                'Time_of_Day': [time_of_day_name],
                                'Temperature': [temp_float],
                                'Weather_Condition': [weather_condition],
                                'Holiday': [holiday_status],
                                'University_Event': [university_event_status]
                            })
                            
                            # Ensure data types match what model expects
                            input_data = input_data.astype({
                                'Month': 'object',
                                'Day': 'int64',
                                'Day_of_Week': 'object',
                                'Time_of_Day': 'object',
                                'Temperature': 'float64',
                                'Weather_Condition': 'object',
                                'Holiday': 'object',
                                'University_Event': 'object'
                            })
                            
                            print(f"Making prediction with features: {input_data.columns.tolist()}")
                            print(f"Data types:\n{input_data.dtypes}")
                            print(f"Input data:\n{input_data}")
                            result = model.predict(input_data)[0]
                            prediction = f"Predicted iced coffee demand: {result:.2f} cups"
                            print(f"Prediction result: {prediction}")
                            
                            # Save prediction to CSV
                            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            row = [
                                timestamp,
                                month_name,
                                day_int,
                                year_int,
                                day_of_week_name,
                                time_of_day_name,
                                temp_float,
                                weather_condition,
                                holiday_status,
                                university_event_status,
                                round(result, 2)
                            ]
                            
                            with open(CSV_FILE, 'a', newline='') as f:
                                writer = csv.writer(f)
                                writer.writerow(row)
                            
                            print(f"Prediction saved to {CSV_FILE}")
                    
                    except ValueError as ve:
                        error = f"Input error: {str(ve)}"
                        print(f"ValueError: {ve}")
                    
        except Exception as e:
            error = f"Error: {str(e)}"
            print(f"Exception: {e}")
            import traceback
            traceback.print_exc()
    
    return render_template('index.html', prediction=prediction, error=error)

@app.route('/download_predictions')
def download_predictions():
    """Download the predictions CSV file"""
    if os.path.exists(CSV_FILE):
        return send_file(CSV_FILE, as_attachment=True, download_name='predictions.csv')
    else:
        return "No predictions yet", 404

@app.route('/view_predictions')
def view_predictions():
    """View all predictions in a table"""
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        return render_template('predictions.html', predictions=df.to_html(classes='table', index=False))
    else:
        return render_template('predictions.html', predictions="<p>No predictions yet</p>")

if __name__ == '__main__':
    print("Starting on http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=False)