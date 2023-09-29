import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score

def load_data(file_path):
    data = pd.read_csv(file_path)
    data_cleaned = data.drop(columns=["address", "full_address", "updated_date", "link"])
    return data_cleaned

def preprocess_data(data):
    true_numerical_cols = ["latitude", "longitude", "latitude_mrt", "longitude_mrt", 
                           "Distance_to_Nearest_MRT_km", "Walking_Time_to_Nearest_MRT_min", 
                           "room_size_sqft"]
    
    # Initialize the scaler
    scaler = StandardScaler()
    
    # Fit and transform the scaler to normalize the numerical features
    data[true_numerical_cols] = scaler.fit_transform(data[true_numerical_cols])
    
    one_hot_cols = ["region", "unit_type", "room_type"]
    ordinal_cols = ["planning_area", "Nearest_MRT_Station"]
    
    # Initialize encoders
    one_hot_encoder = OneHotEncoder(drop="first", sparse=False)
    ordinal_encoder = OrdinalEncoder()
    
    # Apply ordinal encoding to the specified columns
    data[ordinal_cols] = ordinal_encoder.fit_transform(data[ordinal_cols])
    
    # Apply one-hot encoding to the specified columns
    one_hot_encoded = one_hot_encoder.fit_transform(data[one_hot_cols])
    one_hot_encoded_df = pd.DataFrame(one_hot_encoded, columns=one_hot_encoder.get_feature_names_out(one_hot_cols))
    
    # Concatenate the one-hot encoded dataframe with the original dataframe
    data = pd.concat([data, one_hot_encoded_df], axis=1)
    
    # Drop the original categorical columns
    data.drop(one_hot_cols, axis=1, inplace=True)
    
    # Save the scaler and encoder for later use
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(one_hot_encoder, 'one_hot_encoder.pkl')
    joblib.dump(ordinal_encoder, 'ordinal_encoder.pkl')
    
    return data

def split_data(data):
    X = data.drop(["price","status"], axis=1)
    y = data["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X, y, X_train, X_test, y_train, y_test

def train_random_forest(X_train, y_train):
    rf = RandomForestRegressor(
        max_depth=36,
        max_samples=0.8898762217860171,
        max_features='log2',
        min_samples_leaf=1,
        min_samples_split=2,
        n_estimators=1401,
        random_state=42
    )
    rf.fit(X_train, y_train)
    joblib.dump(rf, 'rf_model.pkl')
    return rf

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse**0.5
    r2 = r2_score(y_test, y_pred)
    return rmse, r2

def feature_importance_graph(model,X):
    feature_importance = model.feature_importances_
    feature_names = X.columns
    sorted_idx = feature_importance.argsort()

    plt.figure(figsize=(12, 8))
    plt.barh(feature_names[sorted_idx], feature_importance[sorted_idx])
    plt.xlabel("Feature Importance")
    plt.title("Feature Importance using Random Forest")
    plt.show()

def scatter_plot(y,y_test,X_test,model):
    y_pred = model.predict(X_test)
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r', lw=2)
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title("Actual vs Predicted Rental Prices")
    plt.xlim(0,10000)
    plt.ylim(0,10000)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def main():
    data_path = "Rental_data_final.csv"
    data = load_data(data_path)
    data_preprocessed = preprocess_data(data)
    X, y, X_train, X_test, y_train, y_test = split_data(data_preprocessed)
    rf_model = train_random_forest(X_train, y_train)
    rmse, r2 = evaluate_model(rf_model, X_test, y_test)
    print(f"RMSE: {rmse}, R2: {r2}")
    feature_importance_graph(rf_model,X)
    scatter_plot(y,y_test,X_test,rf_model)

if __name__ == "__main__":
    main()
