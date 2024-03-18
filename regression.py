import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import numpy as np
from data_processing import preprocess_data
import pickle
from config import regions, target


def get_regression_predictions_for_region_and_target(df, region_code, target):

    # Filter the DataFrame for the current region
    df_region = df[df['Country Code'] == region_code]

    # Only use years when target variable is not null
    df_region = df_region.dropna(subset=[target])

    if df_region[target].notnull().sum() < 20:
        return None
    
    # Find features that have no NaN values in rows where target is not null
    features = [col for col in df_region.columns if df_region[col].notnull().all()]

    # Remove the 'Country Code' and 'Year' from features if present, along with the target variable itself
    features = [f for f in features if f not in ['Country Code', 'Year', target]]

    # Split the data into features (X) and targets (Y)
    X = df_region[features]
    Y = df_region[target]

    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Manually split the dataset into training and test sets to preserve time order
    split_index = int(len(X_scaled) * 0.8)
    X_train_full, X_test = X_scaled.iloc[:split_index], X_scaled.iloc[split_index:]
    Y_train_full, Y_test = Y.iloc[:split_index], Y.iloc[split_index:]

    # Initialize TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)
    selected_features_over_folds = []

    for train_index, _ in tscv.split(X_train_full):
        X_train, y_train = X_train_full.iloc[train_index], Y_train_full.iloc[train_index]

        # Initialize the Random Forest model
        model = RandomForestRegressor(n_estimators=200, random_state=42)

        # Apply RFE for feature selection
        selector = RFE(model, n_features_to_select=5, step=1)
        X_train_selected = selector.fit_transform(X_train, y_train)

        # Record the selected features for this fold
        selected_features = X_train_full.columns[selector.support_]
        selected_features_over_folds.append(selected_features)

    # Identify consistently selected features across folds
    consistent_features = list(set.intersection(*map(set, selected_features_over_folds)))

    # Fit the final model on the full training set using consistently selected features
    X_train_final = X_train_full[consistent_features]
    model.fit(X_train_final, Y_train_full)

    # Predict on the test set using the final model
    X_test_final = X_test[consistent_features]
    y_predict = model.predict(X_test_final)

    # Evaluate the final model
    final_r2 = r2_score(Y_test, y_predict)
    
    # Prepare the final DataFrame for return
    df_predictions = {
        'y_real': Y.values,
        'y_predict': model.predict(X_scaled[consistent_features]),
    }

    # Add top feature columns to the predictions DataFrame for plotting
    for feature in consistent_features:
        df_predictions[feature] = X[feature].values

    return df_predictions, final_r2, consistent_features


def compute_and_store_models(df, regions, target):
    results = {}
    for region_code, region_name in regions.items():
        df_predictions, mean_performance, top_features = get_regression_predictions_for_region_and_target(df, region_code, target)
        print(f"{region_name}: {mean_performance:.2f}")
        results[region_name] = {
            'mean_performance': mean_performance,
            'top_features': [feature for feature in top_features],
            'predictions': df_predictions
        }

    # Serialize the results to a file
    with open('model_results.pkl', 'wb') as file:
        pickle.dump(results, file)


if __name__ == '__main__':
    # Load data
    #all_data = pd.read_csv('WDIData.csv', index_col=None)
    all_data = pd.read_csv('climate_data.csv', index_col=None) #truncated data for ease of sharing on github
    df = preprocess_data(all_data, regions, target)
    compute_and_store_models(df, regions, target)
             
    
