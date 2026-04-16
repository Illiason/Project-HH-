# Heart Disease Prediction - Web UI with Model Selection
# Run: python app.py
# Open: http://localhost:5000
# HTML template is in: templates/index.html


# os = library for interacting with the operating system
# We use it for tasks like navigating file directories, managing file paths, and environment variables
import os

# numpy = library for working with numbers, arrays, and math operations
# We use it for numerical calculations
# "as np" = shortcut so we type np instead of numpy
import numpy as np

# joblib = library for efficiently saving and loading large data objects and machine learning models
# We use it to persist Python objects to disk
import joblib


from flask import Flask, request, render_template


# Load pre-trained models from disk (no retraining on startup)

MODELS_DIR = 'models'

print("Loading models from disk...")

xgb_model = joblib.load(os.path.join(MODELS_DIR, 'xgb_model.joblib'))
rf_model   = joblib.load(os.path.join(MODELS_DIR, 'rf_model.joblib'))
lgr_model  = joblib.load(os.path.join(MODELS_DIR, 'lgr_model.joblib'))
accuracies = joblib.load(os.path.join(MODELS_DIR, 'accuracies.joblib'))

MODELS = {
    'xgboost':  {'model': xgb_model,  'name': 'XGBoost',            'acc': accuracies['xgboost']},
    'rf':       {'model': rf_model,   'name': 'Random Forest',       'acc': accuracies['rf']},
    'logistic': {'model': lgr_model,  'name': 'Logistic Regression', 'acc': accuracies['logistic']},
}

for key, info in MODELS.items():
    print(f"  {info['name']}: {info['acc']*100:.1f}%")
print("All models ready!")


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def predict():
    result = None
    probability = None
    selected_model = 'xgboost'
    values = {'age': '50', 'sex': '1', 'cp': '1', 'thalach': '150',
              'oldpeak': '0', 'exang': '0', 'chol': '200', 'slope': '1'}

    if request.method == 'POST':
        if request.form.get('select_model'):
            selected_model = request.form.get('select_model')
        elif request.form.get('selected_model'):
            selected_model = request.form.get('selected_model')

        if request.form.get('action') == 'predict':
            values = {key: request.form.get(key, values[key]) for key in values}
            features = np.array([[
                float(values['cp']), float(values['thalach']),
                float(values['oldpeak']), float(values['exang']),
                float(values['chol']), float(values['slope']),
                float(values['age']), float(values['sex'])
            ]])
            model = MODELS[selected_model]['model']
            result = int(model.predict(features)[0])
            proba = model.predict_proba(features)[0]
            probability = f"{proba[result]*100:.1f}"

    # render_template() loads templates/index.html automatically
    return render_template('index.html', result=result, probability=probability,
                           values=values, selected_model=selected_model, models=MODELS)

if __name__ == '__main__':
    print("\nOpen in browser: http://localhost:5000")
    app.run(debug=False, port=5000)
