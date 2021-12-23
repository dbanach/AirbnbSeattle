import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



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


def agregate_categories(df1, df2, df3, cols):
    '''
    function that agregates categories in variables that have lots of categories

    '''

    for col in cols:
        value_c = df1[col].value_counts()
        vals = value_c.index[:5]
        df1[col] = df1[col].where(df1[col].isin(vals), 'other')
        df2[col] = df2[col].where(df2[col].isin(vals), 'other')
        df3[col] = df3[col].where(df3[col].isin(vals), 'other')


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

    # Define size of bar plot
    plt.figure(figsize=(10, 8))
    # Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    # Add chart labels
    plt.title(model_type + 'FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')