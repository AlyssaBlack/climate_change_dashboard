# EcoRisk Insights: Unveiling Environmental Threats & Drivers

## Overview
EcoRisk Insights offers a multi-faceted analysis of environmental risks across different regions, emphasizing the interplay between various factors and their impact on ecological sustainability. It leverages Random Forest Regression to forecast environmental risks from a comprehensive dataset, providing an interactive platform for users to explore the consequences of various environmental factors. This dashboard integrates advanced analytical techniques within a user-friendly Dash interface, facilitating insightful examinations into the causes and effects of environmental risks.

## Features
- **Targeted Risk Analysis:** Allows users to examine specific environmental risks in detail, expanding the analysis beyond energy depletion.
- **Region-Specific Exploration:** Users can conduct analyses tailored to different regions, identifying unique environmental challenges and impacts.
- **Dynamic Factor Analysis:** The dashboard dynamically presents key factors influencing the chosen environmental risk, offering insight into their contributory roles.
- **Predictive Insights:** Utilizes Random Forest Regression for predictions based on historical data, enhanced by R² metrics for assessing model accuracy.
- **SHAP Value Visualization:** Incorporates SHAP (SHapley Additive exPlanations) values to depict the impact of each feature on the target variable, improving interpretability and deepening insights into factor influence.

## Methodology
The analysis is powered by decision trees for regression, with the selection of relevant features through a TimeSeriesSplit method. This approach ensures accuracy by acknowledging the chronological order of data. The final selection of features is based on their consistent impact across various analyses, ensuring the reliability of insights.

The dashboard calculates the coefficient of determination (R²) by setting aside the last 20% of the data set, offering a robust measure of model accuracy in predicting energy depletion trends.

## Installation
To set up the dashboard locally, follow these steps:

1. Clone the repository:
```
git clone https://github.com/AlyssaBlack/climate-change-dashboard.git
```
2. Navigate to the project directory:
```
cd climate-change-economic-damage-dashboard
```
3. Run the Dash app:
```
python app.py
```

## Usage
After starting the app, navigate to http://127.0.0.1:8050/ in your web browser to access the dashboard.

1. **Select an Environmental Risk:** Start by choosing a specific environmental risk for analysis.
2. **Choose a Region:** Select a region for focused analysis on the selected risk.
3. **Investigate Contributing Factors:** Select a factor to understand its effect on the chosen environmental risk.
4. **Analysis of Predictions vs. Real Data:** A scatter plot displays the comparison between actual outcomes and predictions for the selected factor.
5. **Deep Dive with SHAP Values:** Explore the significance and direction of each feature's impact on the risk assessment through a SHAP summary plot.

## Advanced Usage
Interested users can recompute the predictive models to reflect new data or analytical perspectives. Simply execute:
```
python regression.py
```
This script refreshes the model predictions and feature selections, and is not deterministic.
