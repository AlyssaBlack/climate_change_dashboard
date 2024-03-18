import pandas as pd
from config import features_list, target, regions


def preprocess_data(all_data, regions, target):
    # Keep regional data, drop individual countries, drop redundant columns
    df = all_data[all_data['Country Code'].isin(regions.keys())]
    df = df[df['Indicator Name'].isin(features_list+[target])]
    df = df.drop(['Country Name', 'Indicator Code'], axis=1)

    # Melt the DataFrame
    df_melted = df.melt(id_vars=['Country Code', 'Indicator Name'], var_name='Year', value_name='Value')

    # Pivot the melted DataFrame to get the desired structure
    df = df_melted.pivot_table(index=['Country Code', 'Year'], columns='Indicator Name', values='Value').reset_index()

    # Ensure 'Year' is an integer to sort correctly
    df['Year'] = df['Year'].astype(int)

    # Sort by 'Country Code' and 'Year' to ensure data is in correct order
    df = df.sort_values(by=['Country Code', 'Year'])

    return df

