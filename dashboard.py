import dash
import pickle
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
from config import target


def create_dash_app(precomputed_results):
    app = dash.Dash(__name__)
    # Styling HTML
    title_style = {
    'fontFamily': '"Open Sans", sans-serif',  # Primary font Open Sans with fallback to any available sans-serif
    'textAlign': 'center',  # Center the text
    'color': '#2f3e5c',  # Gray text color
    'fontSize': '24px',  # Larger font size
    'margin': '20px 0 10px 0'  # Margin top and bottom for spacing
    }

    label_style = {
    'fontFamily': '"Open Sans", sans-serif',
    'margin': '20px 0 10px 0',  # Margin top and bottom for spacing
    'fontSize': '18px',  # Slightly larger font size for visibility
    'color': '#2f3e5c',  
    }

    dropdown_style = {
    'fontFamily': '"Open Sans", sans-serif',
    'borderRadius': '10px',  # Rounded corners
    'border': 'none', 
    'padding': '0px',
    'margin': '10px 0',  # Margin top and bottom
    'fontSize': '16px',  # Larger font size
    'background-color': '#e6ecf5',  # Light grey background
    'color': '#2f3e5c',  # Dark grey text
    }

    app.layout = html.Div([
        html.H1("Adjusted Savings: Energy Depletion as % of Gross National Income", style=title_style),
        
        html.Label("Select Region:", style=label_style),
        dcc.Dropdown(
            id='region-dropdown',
            options=[{'label': r, 'value': r} for r in list(precomputed_results.keys())],
            value='Arab World',  # Default value
            style=dropdown_style
        ),

        html.Label("Select Feature:", style=label_style),
        dcc.Dropdown(
            id='feature-dropdown',
            style=dropdown_style
        ),

        dcc.Graph(id='regression-graph'),
    ])

    @app.callback(
        [
            Output('feature-dropdown', 'options'),
            Output('feature-dropdown', 'value'),
            Output('regression-graph', 'figure')
        ],
        [Input('region-dropdown', 'value'),
        Input('feature-dropdown', 'value')]
    )
    def update_content(selected_region, selected_feature):
        region_results = precomputed_results[selected_region]
        r2 = region_results['mean_performance']
        fig = go.Figure()

        if region_results:
            top_features = region_results['top_features']
            options = [{'label': feature, 'value': feature} for feature in top_features]
            feature = selected_feature if selected_feature else top_features[0] if top_features else None

            df_predictions = pd.DataFrame(region_results['predictions'])

            if selected_feature and not df_predictions.empty:
                fig.add_trace(go.Scatter(x=df_predictions[feature], y=df_predictions['y_real'],
                                         mode='markers', name='Actual'))
                fig.add_trace(go.Scatter(x=df_predictions[feature], y=df_predictions['y_predict'],
                                         mode='markers', name='Predicted'))

                fig.update_layout(title=f'{selected_region}: Actual vs. Predicted {target} by {feature}',
                                  xaxis_title=feature,
                                  yaxis_title=target)

                fig.add_annotation(x=1, y=0, xref='paper', yref='paper', 
                                  text=f'R²: {r2:.2f}', showarrow=False,
                                  xanchor='right', yanchor='bottom', font=dict(size=12), align='right')

        else:
            options = []
            feature = None
            fig.add_annotation(x=0.5, y=0.5, xref="paper", yref="paper",
                               text="Data not available for this combination.",
                               showarrow=False, font=dict(size=16))

        return options, feature, fig

    return app

if __name__ == '__main__':
    # Load pre-computed results from the pickle file
    with open('model_results.pkl', 'rb') as file:
        precomputed_results = pickle.load(file)

    app = create_dash_app(precomputed_results)
    app.run_server(debug=True)
