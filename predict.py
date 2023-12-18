# predict.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from joblib import dump, load

lead_scoring_model = RandomForestClassifier(n_estimators=100, random_state=42)
sales_forecasting_model = LinearRegression()

def fit_models(X_train_lead, y_train_lead, X_train_sales, y_train_sales):
    lead_scoring_model.fit(X_train_lead, y_train_lead)
    sales_forecasting_model.fit(X_train_sales, y_train_sales)

    # Save models to disk
    dump(lead_scoring_model, 'lead_scoring_model.pkl')
    dump(sales_forecasting_model, 'sales_forecasting_model.pkl')

def predict_lead_score(features):
    return lead_scoring_model.predict(features)

def predict_sales_forecast(features):
    return sales_forecasting_model.predict(features)