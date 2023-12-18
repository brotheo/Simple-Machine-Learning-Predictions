# app.py
from flask import Flask, render_template, request
from predict import fit_models, predict_lead_score, predict_sales_forecast
from joblib import load

app = Flask(__name__)

# Example training data (replace with your actual training data)
X_train_lead = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
y_train_lead = [0, 1]

X_train_sales = [[10.0, 15.0, 20.0], [25.0, 30.0, 35.0]]
y_train_sales = [100, 200]

# Fit models with training data
fit_models(X_train_lead, y_train_lead, X_train_sales, y_train_sales)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Retrieve form data (replace with your form field names)
        feature1 = float(request.form['feature1'])
        feature2 = float(request.form['feature2'])
        feature3 = float(request.form['feature3'])

        # Make predictions
        new_lead_data = [[feature1, feature2, feature3]]
        new_lead_score = predict_lead_score(new_lead_data)
        new_sales_forecast = predict_sales_forecast(new_lead_data)

        # Your decision-making logic here
        decision = "Your decision-making logic goes here."

        return render_template('result.html', feature1=feature1, feature2=feature2, feature3=feature3, lead_score=new_lead_score, sales_forecast=new_sales_forecast, decision=decision)

if __name__ == '__main__':
    app.run(debug=True)

