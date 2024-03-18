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
    'textAlign': 'center',
    'color': '#2f3e5c', 
    'fontSize': '24px',
    'margin': '20px 0 10px 0'
    }

    label_style = {
    'fontFamily': '"Open Sans", sans-serif',
    'margin': '20px 0 10px 0',
    'fontSize': '18px', 
    'color': '#2f3e5c',  
    }

    dropdown_style = {
    'fontFamily': '"Open Sans", sans-serif',
    'borderRadius': '10px',
    'border': 'none', 
    'padding': '0px',
    'margin': '10px 0',
    'fontSize': '16px', 
    'background-color': '#e6ecf5',
    'color': '#2f3e5c',
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
            disabled=True,
            style=dropdown_style
        ),

        dcc.Graph(id='regression-graph', style={'height': '700px'}),
    ])

    # Callback to update feature dropdown options and re-enable it based on the selected region
    @app.callback(
        [
            Output('feature-dropdown', 'options'),
            Output('feature-dropdown', 'value'),
            Output('feature-dropdown', 'disabled')  # Add this output to control the disabled property
        ],
        [Input('region-dropdown', 'value')]
    )
    def update_feature_dropdown(selected_region):
        region_results = precomputed_results[selected_region]
        top_features = region_results['top_features']
        options = [{'label': feature, 'value': feature} for feature in top_features]
        selected_feature = top_features[0] if top_features else None
        return options, selected_feature, False  # Re-enable dropdown


    @app.callback(
            Output('regression-graph', 'figure'),
        [Input('region-dropdown', 'value'),
        Input('feature-dropdown', 'value')]
    )
    def update_content(selected_region, feature):
        region_results = precomputed_results[selected_region]
        r2 = region_results['mean_performance']
        fig = go.Figure()

        if region_results:
            df_predictions = pd.DataFrame(region_results['predictions'])

            if feature and not df_predictions.empty:
                fig.add_trace(go.Scatter(x=df_predictions[feature], y=df_predictions['y_real'],
                                         mode='markers', name='Actual'))
                fig.add_trace(go.Scatter(x=df_predictions[feature], y=df_predictions['y_predict'],
                                         mode='markers', name='Predicted'))

                fig.update_layout(title={
                                    'text': f'{selected_region}: Actual vs. Predicted {target} by {feature}',
                                    'y':0.9,
                                    'x':0.5,
                                    'xanchor': 'center',
                                    'yanchor': 'top'
                                  },
                                  xaxis_title=feature,
                                  yaxis_title=target,
                                  height=700)

                fig.add_annotation(x=1, y=0, xref='paper', yref='paper', 
                                  text=f'R²: {r2:.2f}', showarrow=False,
                                  xanchor='right', yanchor='bottom', font=dict(size=12), align='right')

        else:
            fig.add_annotation(x=0.5, y=0.5, xref="paper", yref="paper",
                               text="Data not available for this combination.",
                               showarrow=False, font=dict(size=16))

        return fig

    return app

if __name__ == '__main__':
    # Load pre-computed results from the pickle file
    with open('model_results.pkl', 'rb') as file:
        precomputed_results = pickle.load(file)

    app = create_dash_app(precomputed_results)
    app.run_server(debug=True)
