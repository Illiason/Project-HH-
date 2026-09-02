# Heart Disease Prediction

A machine learning project that predicts whether a patient has heart disease from
8 clinical measurements, with a Flask web app for interactive predictions.

## Overvie<img width="1085" height="817" alt="image" src="https://github.com/user-attachments/assets/7fd0340b-e8f4-4cbb-afae-cb5e8e5c5579" />


Three classifiers are trained on the same dataset and compared side by side:

| Model               | Test Accuracy |
|----------------------|:---:|
| Random Forest         | 85.3% |
| XGBoost                | 83.7% |
| Logistic Regression   | 82.6% |

All three are validated beyond a single train/test split, using 5-fold
cross-validation, ROC/AUC, and precision-recall curves (see
[cross_validation.ipynb](cross_validation.ipynb)).

## Dataset

`main_dataset.csv` holds 918 patient records with 8 features and a binary
target (`0` = no disease, `1` = has disease):

- `age` - patient age
- `sex` - 1 = male, 0 = female
- `cp` - chest pain type
- `chol` - serum cholesterol
- `thalach` - maximum heart rate achieved
- `exang` - exercise-induced angina (1 = yes, 0 = no)
- `oldpeak` - ST depression induced by exercise
- `slope` - slope of the peak exercise ST segment

## Project structure

```
app.py                        # Flask web app - model selector + prediction UI
templates/index.html          # Prediction form and results page
static/style.css              # Page styling

train_xgboost.py              # Train + evaluate the XGBoost model
train_randomforest.py         # Train + evaluate the Random Forest model
train_logisticregression.py   # Train + evaluate the Logistic Regression model
train_models_main.ipynb       # Walkthrough: training and comparing all 3 models
cross_validation.ipynb        # Cross-validation, ROC/AUC, precision-recall analysis

models/                       # Saved model files (joblib) used by app.py
iterations/                   # Earlier iterations of dataset construction + slides
main_dataset.csv              # Training data
Links                         # Reference material on XGBoost
```

## Running the app

```bash
pip install flask numpy joblib scikit-learn xgboost pandas
python app.py
```

Then open `http://localhost:5000`, pick a model, enter patient values, and
get a prediction with its confidence score.

## Retraining a model

Each `train_*.py` script loads `main_dataset.csv`, trains one model, and
prints its accuracy, confusion matrix, and classification report:

```bash
python train_randomforest.py
```

## License

MIT - see [LICENSE](LICENSE).
