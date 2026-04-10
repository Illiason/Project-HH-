# Heart Disease Prediction - Web UI with Model Selection
# Run: python app.py
# Open: http://localhost:5000
# HTML template is in: templates/index.html

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from flask import Flask, request, render_template

# Train all 3 models on startup
df = pd.read_csv('main_dataset.csv')
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=333)

print("Training all 3 models...")

xgb_model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=123)
rf_model.fit(X_train, y_train)

lgr_model = LogisticRegression(max_iter=1000, random_state=42)
lgr_model.fit(X_train, y_train)

MODELS = {
    'xgboost':  {'model': xgb_model,  'name': 'XGBoost',            'acc': xgb_model.score(X_test, y_test)},
    'rf':       {'model': rf_model,   'name': 'Random Forest',       'acc': rf_model.score(X_test, y_test)},
    'logistic': {'model': lgr_model,  'name': 'Logistic Regression', 'acc': lgr_model.score(X_test, y_test)},
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
