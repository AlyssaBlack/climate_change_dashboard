import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import numpy as np
from data_processing import preprocess_data
import pickle
import json
import shap
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import textwrap
from mpl_toolkits.axes_grid1 import make_axes_locatable


def get_regression_predictions_for_region_and_target(df, region_code, target):
    """
    Generates regression predictions for a given region and target.
    
    Parameters:
        df (DataFrame): The dataset to work with.
        region_code (str): The region code to filter the data.
        target (str): The target variable for regression.
    
    Returns:
        tuple: A tuple containing the dataframe of predictions, the mean performance score, 
               the list of consistently selected features, and the base64-encoded SHAP plot.
    """
    # Filter the DataFrame for the current region
    df_region = df[df['Country Code'] == region_code]
   
    # Only use years when target variable is not null
    df_region = df_region.dropna(subset=[target])

    if df_region.empty:
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
    test_size_ratio = 0.2
    split_index = int(len(X_scaled) * (1-test_size_ratio))

    if split_index >= 25:  # Enough for 5 splits with at least 5 data points each
        n_splits = 5
    elif split_index >= 20:  # Enough for 4 splits
        n_splits = 4
    elif split_index >= 15:  # Enough for 3 splits
        n_splits = 3
    elif split_index >= 9:  # Enough for 2 splits
        n_splits = 2
    else:
        return None

    X_train_full, X_test = X_scaled.iloc[:split_index], X_scaled.iloc[split_index:]
    Y_train_full, Y_test = Y.iloc[:split_index], Y.iloc[split_index:]
    
    if n_splits == 1:
        raise ValueError("Not enough data points for TimeSeriesSplit.")

    # Use the adjusted number of splits in TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=n_splits)
    selected_features_over_folds = []

    for train_index, _ in tscv.split(X_train_full):
        X_train, y_train = X_train_full.iloc[train_index], Y_train_full.iloc[train_index]

        # Initialize the Random Forest model
        model = RandomForestRegressor(n_estimators=100, random_state=42)

        # Apply RFE for feature selection
        features_to_select = max(1, int(len(features) * 0.3))
        selector = RFE(model, n_features_to_select=features_to_select, step=1)
        X_train_selected = selector.fit_transform(X_train, y_train)

        # Record the selected features for this fold
        selected_features = X_train_full.columns[selector.support_]
        selected_features_over_folds.append(selected_features)

    # Identify consistently selected features across folds
    consistent_features = list(set.intersection(*map(set, selected_features_over_folds)))
    if consistent_features == []:
        return None

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

    shap_plot_base64 = get_shap(model, X_scaled[consistent_features])

    return df_predictions, final_r2, consistent_features, shap_plot_base64


def get_shap(model, X):
    # Create a SHAP summary plot
    # This library doesn't play nice with plotly. We'll save the plot to a buffer and return the base64 string
    buf = BytesIO()
    explainer = shap.Explainer(model) #newer version of shap chooses best explainer
    shap_values = explainer(X)

    # Generate summary plot
    plt.figure(figsize=(12, 8)) #tried (10,8)
    shap.summary_plot(shap_values.values, features=X, feature_names=X.columns, show=False, color_bar=False)

    ax = plt.gca()
    labels = ax.get_yticklabels()
    wrapped_labels = [textwrap.fill(label.get_text(), width=30) for label in labels]
    ax.set_yticklabels(wrapped_labels)
    plt.tight_layout()
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.1)
    plt.rcParams.update({
        'font.size': 6
    })

    # Customize font sizes for the plot here
    plt.title('Feature Impact: Understanding Risk Contributors', fontsize=8)
    plt.xlabel('Contribution to Prediction (SHAP Value)', fontsize=6)
    plt.ylabel('Features', fontsize=6)

    # Create an axis on the right side of `ax`. The width of `cax` can be controlled by `fraction`.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    cbar = plt.colorbar(cax=cax)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(size=0)  
    cbar.ax.set_yticklabels([]) 

    cbar.ax.text(0.5, 1.01, 'High', ha='center', va='bottom', fontsize=6, transform=cbar.ax.transAxes)
    cbar.ax.text(0.5, -0.01, 'Low', ha='center', va='top', fontsize=6, transform=cbar.ax.transAxes)
    cbar.ax.text(1.3, 0.5, 'Feature Importance', rotation=270, fontsize=6, ha='center', va='center', transform=cbar.ax.transAxes)

    # Adjust tick parameters
    ax.tick_params(axis='x', labelsize=6)  # Adjust x-axis tick font size
    ax.tick_params(axis='y', labelsize=6)  # Adjust y-axis tick font size
    plt.savefig(buf, format="png", bbox_inches='tight', dpi=400)
    buf.seek(0)
    summary_plot_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    plt.close()
    return summary_plot_base64


def compute_and_store_models(df, regions, potential_targets):
    results = {}
    for target in potential_targets:
        results[target] = {}
        for region_code, region_name in regions.items():
            result = get_regression_predictions_for_region_and_target(df, region_code, target)
            if result is not None:
                df_predictions, mean_performance, top_features, shap_plot_base64 = result
                # Only save results if mean_performance is within a valid range
                if -1 < mean_performance < 1:
                    print(f"{target}, {region_name}: {mean_performance:.2f}")
                    results[target][region_name] = {
                        'mean_performance': mean_performance,
                        'top_features': top_features,
                        'predictions': df_predictions,
                        'shap_plot': shap_plot_base64
                    }
                else:
                    print(f"{target}, {region_name}: Model performance not valid (r2={mean_performance:.2f}), skipping.")
        if results[target] == {}:
            del results[target]
    # Serialize the results to a file
    with open('model_results.pkl', 'wb') as file:
        pickle.dump(results, file)


if __name__ == '__main__':
    with open('config.json', 'r') as f:
        config = json.load(f)
    regions = config['regions']
    potential_targets = config['potential_targets']

    # Load data
    all_data = pd.read_csv('WDIData.csv', index_col=None)
    #all_data = pd.read_csv('climate_data.csv', index_col=None) #truncated data for ease of sharing on github
    df = preprocess_data(all_data, regions)
    compute_and_store_models(df, regions, potential_targets)
             