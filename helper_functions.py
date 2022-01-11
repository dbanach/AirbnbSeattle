"""
This file contains helper functions that are used in the creation of the model

"""
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline



def change_to_boolean(df, cols):
    '''
    function that receives dataframe df and columns where to change 't' to 1 and 'f' to 0
    cols = list of columns
    '''
    for col in cols:
        df.loc[:, col] = df[col].apply(lambda s: 1 if s == 't' else 0)

def change_from_dollar(df, cols):
    '''
    function that receives a dataframe df and columns where to change from price string format "$1,000.00" to float
    '''

    for col in cols:
        df.loc[:, col] = df[col].apply(lambda s: s if s is np.nan else float(s.replace(',', '')[1:]))

def change_from_percentage(df, cols):
    '''
    function that receives a dataframe df and columns where to change from string format '34%' to float 34
    '''
    for col in cols:
        df.loc[:, col] = df[col].apply(lambda s: s if s is np.nan else float(s.replace('%', '')))

def transform_and_return(df, features, model):
    '''
    Function that receives a dataframe, its features and an already fitted transformer and applies the
    transformer and returns the output as a dataframe
    '''

    df = model.transform(df)
    df = pd.DataFrame(df, columns=features)
    df.reset_index(drop=True, inplace=True)
    return df

def agregate_categories(df1, df2, cols):
    '''
    function that takes a feature that has too many categories and aggregates the lower categories
    into category 'other'. Does that for all dataframes
    '''

    for col in cols:
        value_c = df1[col].value_counts()
        vals = value_c.index[:5]
        df1[col] = df1[col].where(df1[col].isin(vals), 'other')
        df2[col] = df2[col].where(df2[col].isin(vals), 'other')

def dataframe_with_dummies(df, dummies, encoder, features):
    """
    function that receives df dataframe with the independent features and does OHE transformation with
    the already fitted encoder "encoder" and returns a dataframe with all of the OHE features, the rest of the features
    of the df dataframe

    """
    df_dummies = pd.DataFrame(data=encoder.transform(df[dummies]).toarray(), columns=encoder.get_feature_names(dummies))
    df = pd.concat([df[features], df_dummies], axis=1)
    df.columns = features + list(encoder.get_feature_names(dummies))
    return df

def plot_feature_importance(importance, names, model_type, num_features):
    # Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    # Create a DataFrame using a Dictionary
    data = {'feature_names': feature_names, 'feature_importance': feature_importance}
    fi_df = pd.DataFrame(data)

    # Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)
    fi_df = fi_df.iloc[:num_features, :num_features]

    return list(fi_df['feature_names'])

def get_best_features(importance, names, model_type, num_features):
    # Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    # Create a DataFrame using a Dictionary
    data = {'feature_names': feature_names, 'feature_importance': feature_importance}
    fi_df = pd.DataFrame(data)

    # Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)
    fi_df = fi_df.iloc[:num_features, :num_features]

    return list(fi_df['feature_names'])

def distance_latlon(point1, point2):
    '''
    Function that receives the latitude and longitude of 2 points (point1 and point2 are tuples) and returns
    the distance in kilometers
    '''

    return geodesic(point1, point2).km

# helper functions to retrive new data to add
def get_data_to_add(listing, avg_per_period, summer, weekend):
    '''
    Function that receives listing dataframe and with the information in the avg_per_period data frame, converts the
    data in listing to that of the period of summer and weekend that could be 1 or 0 each.
    Then returns the new data to add to the dataset
    '''

    features = list(listing.columns)
    my_listing = listing.copy()
    my_listing.drop(columns=['price'], inplace=True)
    my_listing.summer = summer
    my_listing.weekend = weekend

    add_listing = avg_per_period[(avg_per_period.summer == summer) & (avg_per_period.weekend == weekend)]
    add_listing.drop(columns=['summer', 'weekend', 'avg_price'], inplace=True)

    my_listing = my_listing.merge(add_listing, left_on='id', right_on='listing_id', how='left')
    my_listing = my_listing[~my_listing.price.isna()]
    my_listing = my_listing[features]
    return my_listing

def get_data_all_period(listing, avg_per_period):
    '''
    Function that for one listing dataframe returns the listing, having added information from same listings in
    different periods
    '''
    my_listing = listing.copy()
    for pair in [[1, 1], [1, 0], [0, 1]]:
        my_listing = pd.concat([my_listing, get_data_to_add(listing, avg_per_period, pair[0], pair[1])],
                               ignore_index=True)

    return my_listing

# to impute later on
mostf_filler = Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='most_frequent'))
])

mean_filler = Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='mean'))
])