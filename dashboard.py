import dash
import pickle
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import pandas as pd
import textwrap


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
    'margin': '20px 10px 10px 0',
    'fontSize': '18px', 
    'color': '#2f3e5c',  
    }

    dropdown_style = {
    'fontFamily': '"Open Sans", sans-serif',
    'borderRadius': '10px',
    'border': 'none', 
    'padding': '0px',
    'margin': '10px 10px',
    'fontSize': '16px', 
    'background-color': '#e6ecf5',
    'color': '#2f3e5c',
    #'boxShadow': '5px 8px 5px rgba(0,0,0,0.1)',
    }

    dialogue_style = {
    'textAlign': 'center',
    'color': '#2f3e5c',
    'fontSize': '18px',
    'fontFamily': '"Open Sans", sans-serif',
    'margin': '20px 20px',
    'lineHeight': '1'
    }

    container_style = {
    'backgroundColor': '#f2f5f8',
    'padding': '20px',
    'borderRadius': '30px',
    'boxShadow': '5px 8px 5px rgba(0,0,0,0.1)',
    'margin': '20px 20px'
}

    app.layout = html.Div([

        html.Div([
            html.H1("EcoRisk Insights: Unveiling Environmental Threats & Drivers", style=title_style),
            html.Label("Select Environmental Risk for Analysis:", style=label_style),
            dcc.Dropdown(
                id='target-dropdown',
                options=[{'label': target, 'value': target} for target in precomputed_results.keys()],
                value=next(iter(precomputed_results)),  # Default value
                style=dropdown_style
            ),
            html.Label("Choose a Region: Explore Risk", style=label_style),
            dcc.Dropdown(id='region-dropdown', style=dropdown_style),

            html.Label("Investigate Contributing Factors", style=label_style),
            dcc.Dropdown(id='feature-dropdown', disabled=True, style=dropdown_style),
        ], style=container_style),

        dcc.Graph(id='regression-graph', style={'height': '700px'}),

        html.Div(
        [
            html.P(
                "Now, let's explore in depth the drivers behind environmental risk. The visualization below reveals the influence of "\
                "various factors on our chosen environmental concern, illuminating the key elements that exacerbate or mitigate these risks. "\
                "Understanding these relationships is fundamental for developing targeted, effective strategies for environmental protection "\
                "and sustainability", style=dialogue_style),
            html.Br(),
            html.P(
                "In simpler terms, this chart helps us see which factors matter most. Each dot is a piece of data from our study. "\
                "Where a dot sits left or right shows whether that factor increases or decreases the risk, and how strongly. The "\
                "color of the dots—from blue (lower values) to red (higher values)—indicates the factor's magnitude in our data. "\
                "This way, we can easily spot which aspects have the most significant impact on the environment, guiding us towards"\
                " areas where our efforts can make the most difference.", style=dialogue_style),
            html.Hr(style={'borderTop': '1.5px solid #b0b0b0', 'width': '80%', 'margin': '20px auto'}),
        ], style=container_style
    ),
        html.Div(
            html.Img(
                id='shap-summary-plot', 
                style={
                    'height': 'auto', 
                    'width': '100%', 
                    'object-fit': 'contain', 
                    'max-width': '1500px',
                    'margin-left': 'auto', 
                    'margin-right': 'auto'
                }
            ),
            style={'text-align': 'center'}
        )
    ])

    @app.callback(
        [
            Output('region-dropdown', 'options'),
            Output('region-dropdown', 'value'),
        ],
        [Input('target-dropdown', 'value')]
    )
    def update_region_dropdown(selected_target):
        # Assuming the regions are consistent across all targets, but you may adjust logic as needed
        regions_options = [{'label': r, 'value': r} for r in precomputed_results[selected_target].keys()]
        selected_region = next(iter(precomputed_results[selected_target].keys()))
        return regions_options, selected_region

    # Callback to update feature dropdown options and re-enable it based on the selected region
    @app.callback(
        [
            Output('feature-dropdown', 'options'),
            Output('feature-dropdown', 'value'),
            Output('feature-dropdown', 'disabled')
        ],
        [Input('target-dropdown', 'value'),
        Input('region-dropdown', 'value')]
    )
    def update_feature_dropdown(selected_target, selected_region):
        if selected_region and selected_target:
            top_features = precomputed_results[selected_target][selected_region]['top_features']
            options = [{'label': feature, 'value': feature} for feature in top_features]
            selected_feature = top_features[0] if top_features else None
            return options, selected_feature, False
        else:
            return [], None, True

    @app.callback(
    Output('shap-summary-plot', 'src'),
    [
        Input('target-dropdown', 'value'),
        Input('region-dropdown', 'value')
    ]
    )
    def update_shap_plot(selected_target, selected_region):
        if selected_target in precomputed_results and selected_region in precomputed_results[selected_target]:
            shap_plot_base64 = f"data:image/png;base64,{precomputed_results[selected_target][selected_region]['shap_plot']}"
            return shap_plot_base64
        return None  # Handle case where the plot can't be updated due to invalid selections

    @app.callback(
    Output('regression-graph', 'figure'),
    [
        Input('target-dropdown', 'value'),
        Input('region-dropdown', 'value'),
        Input('feature-dropdown', 'value')
    ]
    )
    def update_content(selected_target, selected_region, selected_feature):
        region_results = precomputed_results[selected_target][selected_region]
        r2 = region_results['mean_performance']
        fig = go.Figure()

        if region_results:
            df_predictions = pd.DataFrame(region_results['predictions'])

            if selected_feature and not df_predictions.empty:
                wrapped_title = "<br>".join(textwrap.wrap(f"{selected_region}: Actual vs. Predicted {selected_target} by {selected_feature}", width=500))

                fig.add_trace(go.Scatter(x=df_predictions[selected_feature], y=df_predictions['y_real'],
                                         mode='markers', name='Actual'))
                fig.add_trace(go.Scatter(x=df_predictions[selected_feature], y=df_predictions['y_predict'],
                                         mode='markers', name='Predicted'))

                fig.update_layout(title={
                                    'text': wrapped_title,
                                    'y':0.9,
                                    'x':0.5,
                                    'xanchor': 'center',
                                    'yanchor': 'top'
                                  },
                                  xaxis_title=selected_feature,
                                  yaxis_title=selected_target,
                                  height=700,
                                  legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.5)'),  # Adjust legend position
                                )

                fig.add_annotation(x=1, y=0, xref='paper', yref='paper', 
                                  text=f'R²: {r2:.2f}', showarrow=False,
                                  xanchor='right', yanchor='bottom', font=dict(size=12), align='right')

        return fig

    return app

if __name__ == '__main__':
    # Load pre-computed results from the pickle file
    with open('model_results.pkl', 'rb') as file:
        precomputed_results = pickle.load(file)

    app = create_dash_app(precomputed_results)
    app.run_server(debug=True)
