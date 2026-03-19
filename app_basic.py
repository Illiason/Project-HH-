# Basic Heart Disease Prediction - Skeleton Web UI
# Run: python app_basic.py
# Open: http://localhost:5000

# pandas = library for reading CSV files and working with tables
import pandas as pd
# numpy = library for working with numbers and arrays
import numpy as np
# train_test_split = splits data into training (80%) and testing (20%)
from sklearn.model_selection import train_test_split
# XGBClassifier = XGBoost machine learning model for yes/no predictions
from xgboost import XGBClassifier
# Flask = library that creates a web server (handles browser requests)
# request = gets data that user typed in the form
# render_template_string = fills HTML template with Python values
from flask import Flask, request, render_template_string

# --- Train model when the script starts ---

# Read our 918-row dataset from CSV file
df = pd.read_csv('main_dataset.csv')
# X = 8 features (everything except target) - what we know about the patient
X = df.drop('target', axis=1)
# y = target column - what we want to predict (0=healthy, 1=disease)
y = df['target']
# Split: 80% for training (734 rows), 20% for testing (184 rows)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
# Create XGBoost model with 100 trees, max 5 levels deep, learning rate 0.1
model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=123, eval_metric='logloss')
# Train the model - it learns patterns from 734 patients
model.fit(X_train, y_train)
# Test accuracy - how many of 184 test patients did it predict correctly
accuracy = model.score(X_test, y_test)

# Create Flask web application
app = Flask(__name__)

# HTML template - this is the web page the user sees in the browser
# {{ }} = Jinja2 syntax - Flask replaces these with Python values
# {% %} = Jinja2 logic - if/else statements inside HTML
HTML = """
<html>
<head><title>Heart Disease Prediction</title></head>
<body>
<h1>Heart Disease Prediction</h1>
<hr>

<!-- form method="POST" = when user clicks Predict, send data to server -->
<form method="POST">

  <!-- Age input: type="number" = only accepts numbers -->
  <!-- name="age" = Flask uses this name to get the value -->
  <!-- value="{{ v.age }}" = keeps the value after form submission -->
  <label>Age:</label>
  <input type="number" name="age" value="{{ v.age }}"><br><br>

  <!-- Sex dropdown: select = dropdown menu -->
  <!-- 'selected' if v.sex=='1' = remembers what user chose after submission -->
  <label>Sex:</label>
  <select name="sex">
    <option value="1" {{ 'selected' if v.sex=='1' }}>Male</option>
    <option value="0" {{ 'selected' if v.sex=='0' }}>Female</option>
  </select><br><br>

  <!-- Chest Pain Type: 4 options from Typical Angina to Asymptomatic -->
  <label>Chest Pain Type (cp):</label>
  <select name="cp">
    <option value="1" {{ 'selected' if v.cp=='1' }}>1 - Typical Angina</option>
    <option value="2" {{ 'selected' if v.cp=='2' }}>2 - Atypical Angina</option>
    <option value="3" {{ 'selected' if v.cp=='3' }}>3 - Non-Anginal Pain</option>
    <option value="4" {{ 'selected' if v.cp=='4' }}>4 - Asymptomatic</option>
  </select><br><br>

  <!-- Max Heart Rate: number input for beats per minute -->
  <label>Max Heart Rate (thalach):</label>
  <input type="number" name="thalach" value="{{ v.thalach }}"><br><br>

  <!-- ST Depression: step="0.1" = allows decimal numbers like 1.5, 2.3 -->
  <label>ST Depression (oldpeak):</label>
  <input type="number" name="oldpeak" step="0.1" value="{{ v.oldpeak }}"><br><br>

  <!-- Exercise Angina: simple Yes/No dropdown -->
  <label>Exercise Angina (exang):</label>
  <select name="exang">
    <option value="0" {{ 'selected' if v.exang=='0' }}>No</option>
    <option value="1" {{ 'selected' if v.exang=='1' }}>Yes</option>
  </select><br><br>

  <!-- Cholesterol: number input in mg/dl -->
  <label>Cholesterol (chol):</label>
  <input type="number" name="chol" value="{{ v.chol }}"><br><br>

  <!-- ST Slope: 3 options from healthy to dangerous -->
  <label>ST Slope:</label>
  <select name="slope">
    <option value="1" {{ 'selected' if v.slope=='1' }}>1 - Upsloping</option>
    <option value="2" {{ 'selected' if v.slope=='2' }}>2 - Flat</option>
    <option value="3" {{ 'selected' if v.slope=='3' }}>3 - Downsloping</option>
  </select><br><br>

  <!-- Submit button - sends form data to server -->
  <button type="submit">Predict</button>
</form>

<!-- Show result only if user clicked Predict (result is not none) -->
{% if result is not none %}
<hr>
<!-- If result=1 show "HAS HEART DISEASE", if result=0 show "NO HEART DISEASE" -->
<h2>Result: {{ 'HAS HEART DISEASE' if result == 1 else 'NO HEART DISEASE' }}</h2>
<!-- Show probability percentage (e.g., 85.3%) -->
<p>Probability: {{ probability }}%</p>
{% endif %}
</body>
</html>
"""

# @app.route('/') = when someone visits http://localhost:5000, run this function
# methods=['GET', 'POST'] = handle both page load (GET) and form submission (POST)
@app.route('/', methods=['GET', 'POST'])
def predict():
    # result = prediction (0 or 1), starts as None (no prediction yet)
    result = None
    # probability = confidence percentage, starts as None
    probability = None
    # v = default values shown in the form when page first loads
    v = {'age': '50', 'sex': '1', 'cp': '1', 'thalach': '150', 'oldpeak': '0', 'exang': '0', 'chol': '200', 'slope': '1'}

    # If user clicked "Predict" button (POST request)
    if request.method == 'POST':
        # Get all values the user typed/selected in the form
        v = {key: request.form[key] for key in v}

        # Convert form values to numbers in a numpy array
        # Order MUST match training data: cp, thalach, oldpeak, exang, chol, slope, age, sex
        # [[...]] = 2D array because model expects a table (even for 1 patient)
        features = np.array([[float(v['cp']), float(v['thalach']), float(v['oldpeak']), float(v['exang']), float(v['chol']), float(v['slope']), float(v['age']), float(v['sex'])]])

        # model.predict() = get prediction: 0 (no disease) or 1 (has disease)
        # [0] = get first (and only) result from the array
        result = int(model.predict(features)[0])

        # model.predict_proba() = get probability for each class
        # Returns [probability_of_0, probability_of_1]
        # proba[result] = get probability of the predicted class
        proba = model.predict_proba(features)[0]
        # Format as percentage with 1 decimal (e.g., "85.3")
        probability = f"{proba[result]*100:.2f}"

    # render_template_string = fill HTML template with our values and send to browser
    # result, probability, v, accuracy = Python values that replace {{ }} in HTML
    return render_template_string(HTML, result=result, probability=probability, v=v, accuracy=f"{accuracy*100:.2f}")

# This runs when you execute: python app_basic.py
if __name__ == '__main__':
    print("Open: http://localhost:5000")
    # Start the web server on port 5000
    # debug=False = don't show detailed errors (safer)
    # The server keeps running until you press Ctrl+C
    app.run(debug=False, port=5000)
