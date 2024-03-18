Sustainability Dashboard: Energy Depletion Analysis

##Overview
This dashboard offers a comprehensive look into energy resource management across various regions, with a focus on the balance between economic growth and sustainable use of natural resources. It analyzes the impact of energy depletion on a country's Gross National Income (GNI), providing insights into how different areas are navigating the challenges of sustainability. This project utilizes Random Forest Regression to predict economic damages based on historical data and presents the analysis through an intuitive, user-friendly interface built with Dash.

##Features
- **Predictive Modeling:** Utilizes Random Forest Regression to predict economic damages based on key indicators.
- **Dynamic Feature Selection:** The dashboard dynamically updates to display the most relevant features impacting economic damages, allowing for a detailed analysis of the contributing factors.
- **Region-Specific Analysis:** Users can select different regions for a tailored analysis that highlights the unique challenges and economic impacts faced by each area.
- **R² Metric Display:** Provides the R² value of the test set to indicate the model's accuracy and how well the predictive model fits the observed data.

##Methodology
The analysis is powered by decision trees for regression, with the selection of relevant features through a TimeSeriesSplit method. This approach ensures accuracy by acknowledging the chronological order of data. The final selection of features is based on their consistent impact across various analyses, ensuring the reliability of insights.

The dashboard calculates the coefficient of determination (R²) by setting aside the last 20% of the data set, offering a robust measure of model accuracy in predicting energy depletion trends.

##Installation
To set up the dashboard locally, follow these steps:

1. Clone the repository:
```
git clone https://github.com/yourusername/climate-change-economic-damage-dashboard.git
```
2. Navigate to the project directory:
```
cd climate-change-economic-damage-dashboard
```
3. Run the Dash app:
```
python app.py
```

##Usage
After starting the app, navigate to http://127.0.0.1:8050/ in your web browser to access the dashboard.

1. **Select a Region:** Choose a region to focus the analysis on.
2. **Select a Feature:** Choose a feature to visualize how it influences energy depletion.
3. **Analyze Predictions and Real Data:** The dashboard will display a scatter plot comparing actual and predicted values of energy depletion based on the selected feature.

##Advanced Usage
Interested users can recompute the predictive models to reflect new data or analytical perspectives. Simply execute:
```
python regression.py
```
This script refreshes the model predictions and feature selections, and is not deterministic.