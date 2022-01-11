
import pandas as pd
import numpy as np
import time
import datetime
import pickle
from geopy.distance import geodesic
import requests
import json
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error,mean_absolute_percentage_error
from sklearn.model_selection import RandomizedSearchCV
import config as cf
from helper_functions import *

# YELP_DATA = if we'll create data from Yelp Api (True) or we'll load it from a file (False).
YELP_DATA = cf.CREATE_YELP
YELP_RADIUS =  cf.RADIUS_YELP

def preprocessing_listings(listings):
    to_delete = cf.TO_DELETE
    listings.drop(columns=to_delete, inplace=True)

    # ## Changing data type

    # first i'll change features that have values 't' and 'f' to 1 and 0
    to_boolean = cf.TO_BOOLEAN
    change_to_boolean(listings, to_boolean)

    # change features that have % to float
    change_from_percentage(listings, cf.FROM_PERCENTAGE)

    # change features that have format $1000,00 to float
    change_from_dollar(listings, ['price', 'extra_people'])

    # ## Fixing problems with data in BATHROOMS AND BEDROOMS
    # first we'll change the value of the bathrooms where also bedrooms is 0
    my_index = listings[(listings.bathrooms == 0) & (listings.bedrooms == 0)].index
    listings.loc[my_index, 'bathrooms'] = listings[
        (listings.bathrooms == 0) & (listings.bedrooms == 0)].bathrooms.apply(lambda x: np.nan if x == 0 else x)

    # now we'll change all of the values where bedrooms are 0
    my_index = listings[(listings.bedrooms == 0)].index
    listings.loc[my_index, 'bedrooms'] = listings[(listings.bedrooms == 0)].bedrooms.apply(
        lambda x: np.nan if x == 0 else x)

    # ## Dealining with nan's
    for col in cf.TO_FILL_NA:
        listings[col].fillna('NA', inplace=True)

    # there are 2 rows that they dont have info about the host so since there are so few instead of imputing something
    # i'll just drop them
    listings = listings[~listings.host_since.isna()]

    # i'll also drop rows with nan zipcode since there are so little nan rows and i dont want to use later on some demographic data
    # that could be mistaken
    listings = listings[~listings.zipcode.isna()]
    # also eliminate mistaken zipcode value
    listings = listings[~(listings.zipcode == '99\n98122')]

    listings.reset_index(drop=True, inplace=True)
    return listings

def final_preprocessing_listings(listings):
    listings_train, listings_test = train_test_split(listings, test_size=.2, random_state=12)

    # features to fill with mean
    fill_with_mean = cf.FILL_WITH_MEAN
    # features to fill the nans with most frequent
    fill_with_most_frequent = cf.FILL_WITH_MOSTF

    the_rest = list(set(listings.columns) - set(fill_with_mean) - set(fill_with_most_frequent))
    features = fill_with_most_frequent + fill_with_mean + the_rest

    listings_train = listings_train[features]
    listings_test = listings_test[features]

    transformers = ColumnTransformer([
        ('mostf', mostf_filler, fill_with_most_frequent),
        ('mean', mean_filler, fill_with_mean),
        ('the_rest', 'passthrough', the_rest)
    ])

    model = transformers.fit(listings_train)
    listings_train = transform_and_return(listings_train, features, model)
    listings_test = transform_and_return(listings_test, features, model)

    # change data type tu numeric if possible
    for df in [listings_train, listings_test]:
        for col in listings_train.columns:
            try:
                for df in [listings_train, listings_val, listings_test]:
                    df.loc[:, col] = pd.to_numeric(df[col])
            except:
                pass

    return listings_train, listings_test

def preprocessing_calendar(calendar):
    '''
    Function that does the preprocessing of the calendar dataframe and returns it after the changes

    '''

    # changing values in 'available' from 't' and 'f' to 1,0
    change_to_boolean(calendar, ['available'])

    # changing type of 'date' from object to datetime. (i'll need this later on to manipulate this feature)
    calendar['date'] = pd.to_datetime(calendar['date'])

    # change format from price
    change_from_dollar(calendar, ['price'])

    avg_price_per_id = calendar.groupby(calendar.listing_id).mean()
    avg_price_per_id.reset_index(level=0, inplace=True)
    avg_price_per_id = avg_price_per_id[['listing_id', 'price']]
    avg_price_per_id.columns = ['listing_id', 'avg_price']

    ids_to_delete = list(avg_price_per_id[avg_price_per_id.avg_price.isna()].listing_id)
    calendar = calendar[~calendar.listing_id.isin(ids_to_delete)]

    # add avg_price to calendar df
    calendar = calendar.merge(avg_price_per_id, on='listing_id', how='left')

    # change value of price feature if it's nan
    calendar.loc[:, 'price'] = calendar.apply(
        lambda row: row['price'] if not np.isnan(row['price']) else row['avg_price'], axis=1)
    calendar.drop(columns=['avg_price'], inplace=True)

    return calendar

def adjustments(calendar,listings_train,listings_test):
    '''
    Some minor adjustments to the data

    '''

    # i'll eliminate the rows that are below and above 3 times the std of price (not so many rows).
    # i'll only do this on the train set
    mean_price = listings_train.price.mean()
    std_price = listings_train.price.std()
    min_price = mean_price - 3 * std_price
    max_price = mean_price + 3 * std_price
    listings_train = listings_train[(listings_train.price >= min_price) & (listings_train.price <= max_price)]

    # change the order of the features to see price at the end
    rest = list(set(listings_train.columns) - set(['price']))
    features = rest + ['price']
    listings_train = listings_train[features]
    listings_test = listings_test[features]

    right_df = calendar.available.groupby(calendar['listing_id']).apply(lambda x: x.sum() / x.count()).reset_index(
        level=0)
    right_df.columns = ['id', 'occupancy']
    right_df.head()

    listings_train = listings_train.merge(right_df, on='id', how='left')
    listings_test = listings_test.merge(right_df, on='id', how='left')
    features = features + ['occupancy']
    # Fill Na in new feature with mean
    for df in [listings_train, listings_test]:
        df.fillna(listings_train.occupancy.mean(), inplace=True)

    # we'll create features of 'summer', 'weekend'
    for df in [listings_train, listings_test]:
        df['weekend'] = 0
        df['summer'] = 0

    return listings_train,listings_test

def calendar_creation(calendar):
    '''
    function that receives calendar dataframe and creates a few features and returns it
    '''

    # calendar dataframe
    # we'll add some features to the calendar dataframe
    calendar['day_of_week'] = pd.DatetimeIndex(calendar.date).dayofweek
    calendar['month'] = pd.DatetimeIndex(calendar.date).month
    calendar['week'] = pd.DatetimeIndex(calendar.date).week
    calendar['weekend'] = calendar['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    calendar['summer'] = calendar['week'].apply(lambda x: 1 if ((x >= 23) & (x <= 34)) else 0)
    return calendar

def data_creation(calendar, listings_train,listings_test):
    '''
    Function that creates data to add to listings_train/test datasets and returns the dataframes with created data

    '''


    # create dataframe with the price to create new data
    df_avg_per_id_period = pd.DataFrame(calendar.groupby(['listing_id', 'summer', 'weekend']).price.mean()).reset_index(
        level=[0, 1, 2])

    # create dataframe to have the average price in all the combinations (summer=[0,1], weekend =[0,1]) to check if the data is real
    # meaning that it's highly unlikely that the price would be the same in every mommnent so a fast way to filter
    # data that we imputed because it was unavailable it is by filtering listings that have same price than the average
    check_real_data = df_avg_per_id_period.groupby(df_avg_per_id_period.listing_id).price.mean().reset_index(level=0)
    check_real_data.columns = ['listing_id', 'avg_price']

    # merge the data to have a new column in order to see the rows that represent real data and filtering that
    df_avg_per_id_period = df_avg_per_id_period.merge(check_real_data, on='listing_id', how='left')
    df_avg_per_id_period = df_avg_per_id_period[df_avg_per_id_period.price != df_avg_per_id_period.avg_price]

    listings_train = get_data_all_period(listings_train, df_avg_per_id_period)
    listings_test = get_data_all_period(listings_test, df_avg_per_id_period)

    return listings_train, listings_test

def YELP(listings_train, listings_test):
    '''
    function that creates features based on yelp api information about restaurands

    '''

    # i'll add a new feature that's really a copy of zipcode to use in the creation of features with Yelp API
    listings_train['zipcode2'] = listings_train.zipcode
    listings_test['zipcode2'] = listings_test.zipcode

    # ### 6.7.1 - Getting data from API

    if YELP_DATA:
        api_key = cf.API_KEY
        headers = {'Authorization': f'Bearer {api_key}'}
        url = cf.URL
        names = []
        ratings = []
        latitudes = []
        longitudes = []
        prices = []
        zip_codes = []

        # since i saved the data with this already changed i commented this code snippet

        offset = 0
        while offset <= 30000:
            params = {'term': 'Restaurants', 'location': 'seattle', 'limit': 50, 'radious': 10000, 'offset': offset}
            response = requests.get(url, params=params, headers=headers)
            data = response.json()
            n = 0
            while n < 50:
                #         print(n)
                try:
                    restaurant = data['businesses'][n]

                    # first create variable so in case one element is not there it will go to except

                    price = restaurant['price']
                    zipcode = restaurant['location']['zip_code']
                    name = restaurant['name']
                    rating = restaurant['rating']
                    lat = restaurant['coordinates']['latitude']
                    lon = restaurant['coordinates']['longitude']

                    prices.append(price)
                    zip_codes.append(zipcode)
                    names.append(name)
                    ratings.append(rating)
                    latitudes.append(lat)
                    longitudes.append(lon)
                    n += 1
                except:
                    ## some of the data gathered are not going to have the necessary information
                    ## so we skip those
                    n += 1
            offset += 50

        yelpdata = pd.DataFrame(
            {'name': names, 'rating': ratings, 'zipcode': zip_codes, 'prices': prices, 'lat': latitudes,
             'lon': longitudes})
        # we'll drop duplicates just in case
        yelpdata = yelpdata.drop_duplicates(ignore_index=True)

        # we'll change the price from $,$$.. to 1,2..
        # since i saved the data with this already changed i commented this code snippet

        yelpdata.loc[:, 'prices'] = yelpdata.prices.apply(lambda s: len(s))

    else:
        with open(cf.YELPDATA_FILE, "rb") as input_file:
            yelpdata = pickle.load(input_file)

    # ### 6.7.2 - Creation of rating and prices features.

    # the easiest way to use this data is by grouping by zipcode and get the average rating and prices
    rating_df = yelpdata.rating.groupby(yelpdata['zipcode']).mean().reset_index(level=0)
    price_df = yelpdata.prices.groupby(yelpdata['zipcode']).mean().reset_index(level=0)

    rating_price_df = rating_df.merge(price_df, on='zipcode')
    rating_price_df.columns = ['code', 'avg_rating_per_zipcode', 'avg_prices_per_zipcode']
    for df in [listings_train, listings_test]:
        df.loc[:, 'zipcode2'] = df.zipcode2.apply(lambda s: str(s))
    #     df.loc[:,:] = df.merge(rating_price_df,left_on='zipcode2',right_on='zipcode',how='left')
    listings_train = listings_train.merge(rating_price_df, left_on='zipcode2', right_on='code', how='left')
    listings_test = listings_test.merge(rating_price_df, left_on='zipcode2', right_on='code', how='left')

    return listings_train, listings_test

def Yelp_Radius(listings_train, listings_test):
    '''


    '''

    with open(cf.YELPDATA_FILE, "rb") as input_file:
        yelpdata = pickle.load(input_file)

    if YELP_RADIUS:

        RADIUS = 5

        ratings = [[], []]
        prices = [[], []]
        counts = [[], []]

        for num, df in enumerate([listings_train, listings_test]):
            for index, elem in (enumerate(list(df.zipcode))):
                point1 = (df.latitude[index], df.longitude[index])
                my_data = yelpdata.copy()
                my_series = my_data.apply(lambda row: distance_latlon(point1, (row['lat'], row['lon'])), axis=1)
                my_series = my_series[my_series < RADIUS]
                my_index = my_series.index
                my_data = my_data.iloc[my_index]
                ratings[num].append(my_data.rating.mean())
                prices[num].append(my_data.prices.mean())
                counts[num].append(my_data.rating.count())

        pickle.dump(ratings, open("ratings2.pkl", "wb"))
        pickle.dump(prices, open("prices2.pkl", "wb"))
        pickle.dump(counts, open("counts2.pkl", "wb"))

    else:
        with open("ratings2.pkl", "rb") as input_file:
            ratings = pickle.load(input_file)
        with open("prices2.pkl", "rb") as input_file:
            prices = pickle.load(input_file)
        with open("counts2.pkl", "rb") as input_file:
            counts = pickle.load(input_file)

    listings_train['avg_rating_radius'] = pd.Series(ratings[0])
    listings_test['avg_rating_radius'] = pd.Series(ratings[1])

    listings_train['avg_prices_radius'] = pd.Series(prices[0])
    listings_test['avg_prices_radius'] = pd.Series(prices[1])

    listings_train['restaurants_radius'] = pd.Series(counts[0])
    listings_test['restaurants_radius'] = pd.Series(counts[1])

    return listings_train, listings_test

def feature_creation(listings_train,listings_test):
    '''

    function that creates new features in listings_train/test

    '''



    # ##  - Number of confirmations
    listings_train.loc[:, 'num_conf'] = listings_train.host_verifications.apply(lambda s: s.count(',') + 1)
    listings_test.loc[:, 'num_conf'] = listings_test.host_verifications.apply(lambda s: s.count(',') + 1)

    # ##  - Years of host being active

    NOW = 2016
    listings_train.loc[:, 'years_host'] = listings_train.host_since.apply(lambda x: (NOW - int(x[:4])))
    listings_test.loc[:, 'years_host'] = listings_test.host_since.apply(lambda x: (NOW - int(x[:4])))

    listings_train['same_neighbourhood'] = (listings_train.host_neighbourhood == listings_train.neighbourhood)
    listings_train.loc[:, 'same_neighbourhood'] = listings_train.same_neighbourhood.apply(
        lambda x: 1 if x == True else 0)
    listings_test['same_neighbourhood'] = (listings_test.host_neighbourhood == listings_test.neighbourhood)
    listings_test.loc[:, 'same_neighbourhood'] = listings_test.same_neighbourhood.apply(lambda x: 1 if x == True else 0)

    # ##  - Binary features from Amenities

    for df in [listings_train, listings_test]:
        df.loc[:, 'has_ac'] = df.amenities.apply(lambda s: 1 if 'conditioning' in s.lower() else 0)
        df.loc[:, 'has_internet'] = df.amenities.apply(lambda s: 1 if 'internet' in s.lower() else 0)
        df.loc[:, 'has_pool'] = df.amenities.apply(lambda s: 1 if 'pool' in s.lower() else 0)
        df.loc[:, 'has_tv'] = df.amenities.apply(lambda s: 1 if 'tv' in s.lower() else 0)
        df.loc[:, 'smoking_allowed'] = df.amenities.apply(lambda s: 1 if 'smoking' in s.lower() else 0)
        df.loc[:, 'washer'] = df.amenities.apply(lambda s: 1 if 'washer' in s.lower() else 0)
        df.loc[:, 'smoke_detector'] = df.amenities.apply(lambda s: 1 if 'detector' in s.lower() else 0)
        df.loc[:, 'pet_allowed'] = df.amenities.apply(lambda s: 1 if 'pet' in s.lower() else 0)
        df.loc[:, 'kitchen'] = df.amenities.apply(lambda s: 1 if 'kitchen' in s.lower() else 0)
        df.loc[:, 'iron'] = df.amenities.apply(lambda s: 1 if 'iron' in s.lower() else 0)
        df.loc[:, 'fireplace'] = df.amenities.apply(lambda s: 1 if 'fireplace' in s.lower() else 0)
        df.loc[:, 'hot_tub'] = df.amenities.apply(lambda s: 1 if 'tub' in s.lower() else 0)
        df.loc[:, 'heating'] = df.amenities.apply(lambda s: 1 if 'heating' in s.lower() else 0)
        df.loc[:, 'gym'] = df.amenities.apply(lambda s: 1 if 'gym' in s.lower() else 0)
        df.loc[:, 'parking'] = df.amenities.apply(lambda s: 1 if 'parking' in s.lower() else 0)
        df.loc[:, 'elevator'] = df.amenities.apply(lambda s: 1 if 'elevator' in s.lower() else 0)
        df.loc[:, 'doorman'] = df.amenities.apply(lambda s: 1 if 'doorman' in s.lower() else 0)
        df.loc[:, 'breakfast'] = df.amenities.apply(lambda s: 1 if 'breakfast' in s.lower() else 0)

    amenities_list = cf.AMENITIES_LIST
    # ## 6.5 - Listing within downtown or not
    # Checking the Yelp Api , the center of seattle has coordenates :'longitude': -122.3355, 'latitude': 47.6254.
    min_lat = cf.MIN_LAT
    max_lat = cf.MAX_LAT
    min_lon = cf.MIN_LON
    max_lon = cf.MAX_LON

    for df in [listings_train, listings_test]:
        df['is_downtown'] =((df.latitude >= min_lat) & (df.latitude <= max_lat)) &((df.longitude >= min_lon) & (df.longitude <= max_lon))
        df.loc[:,'is_downtown'] = df.is_downtown.apply(lambda s: 1 if s==True else 0)

    # ## 6.6 - Distance of listing to downtown

    CENTER_LAT = cf.DOWNTOWN_LAT
    CENTER_LON = cf.DOWNTOWN_LON
    listings_train['distance_to_downtown'] =((listings_train.latitude - CENTER_LAT) ** 2 + (listings_train.longitude - CENTER_LON) ** 2)**(1/2)
    listings_test['distance_to_downtown'] =((listings_test.latitude - CENTER_LAT) ** 2 + (listings_test.longitude - CENTER_LON) ** 2)**(1/2)

    listings_train, listings_test = YELP(listings_train, listings_test)
    listings_train, listings_test = Yelp_Radius(listings_train, listings_test)

    return listings_train,listings_test


def final_adjustments(listings_train, listings_test):
    '''
    Final adjustments to the train and test set
    '''

    # ## - Bed Type
    # change it to 1 or 0 depending if it is "Real bed" or not

    listings_train.loc[:, 'bed_type'] = listings_train.bed_type.apply(lambda s: 1 if s == 'Real Bed' else 0)
    listings_test.loc[:, 'bed_type'] = listings_test.bed_type.apply(lambda s: 1 if s == 'Real Bed' else 0)

    # ## 7.2 **DELETE** features no longer useful

    to_delete = cf.TO_DELETE_FINAL

    for df in [listings_train, listings_test]:
        for col in to_delete:
            df.drop(columns=[col], inplace=True)

    # ## 7.3 - Inpute values in features created.

    for df in [listings_train, listings_test]:
        for col in ['avg_rating_per_zipcode', 'avg_prices_per_zipcode', 'avg_rating_radius', 'avg_prices_radius']:
            df[col].fillna(listings_train[col].mean(), inplace=True)

    # ## 7.4 - Change categories of features
    for df in [listings_train, listings_test]:
        df.loc[:, 'property_type'] = df.property_type.apply(
            lambda s: s if ((s == 'Apartment') or (s == 'House')) else 'Other')

    # features with 2 many categories we'll aggragate features with not so many data into "other" category.
    # we'll keep only 5 categories including other

    aggregate = cf.AGGREGATE
    agregate_categories(listings_train, listings_test, aggregate)
    listings_train.loc[:, 'neighbourhood'] = listings_train.neighbourhood.apply(lambda s: s if s != 'NA' else 'other')
    listings_test.loc[:, 'neighbourhood'] = listings_test.neighbourhood.apply(lambda s: s if s != 'NA' else 'other')

    # deal with ordinal features

    host_resp_dict = cf.HOST_RESP_DICT
    listings_train.loc[:, 'host_response_time'] = listings_train.host_response_time.apply(lambda x: host_resp_dict[x])
    listings_test.loc[:, 'host_response_time'] = listings_test.host_response_time.apply(lambda x: host_resp_dict[x])

    cancellation_dict = cf.CANCELATION_DICT
    listings_train.loc[:, 'cancellation_policy'] = listings_train.cancellation_policy.apply(
        lambda x: cancellation_dict[x])
    listings_test.loc[:, 'cancellation_policy'] = listings_test.cancellation_policy.apply(
        lambda x: cancellation_dict[x])

    # ##  OneHotEncoding

    for df in [listings_train, listings_test]:
        # making sure all elements in zipcode are strings so there wont be an error doing onehotencoder
        df.loc[:, 'zipcode'] = df.zipcode.apply(lambda s: str(s))

    # onehotencoding the nominal features
    enc = OneHotEncoder(handle_unknown='ignore')
    nominal = ['property_type', 'neighbourhood', 'zipcode', 'room_type']
    enc.fit(listings_train[nominal])
    features_ = list(set(listings_train.columns) - set(nominal))
    listings_train = dataframe_with_dummies(listings_train, nominal, enc, features_)
    listings_test = dataframe_with_dummies(listings_test, nominal, enc, features_)

    return listings_train,listings_test

def split_Xy(listings_train, listings_test):
    # need to split in X,y
    X_train = listings_train.loc[:, listings_train.columns != 'price']
    y_train = listings_train.price
    y_train = y_train.astype('float')

    X_test = listings_test.loc[:, listings_test.columns != 'price']
    y_test = listings_test.price
    y_test = y_test.astype('float')

    return X_train, y_train, X_test, y_test

def get_model_with_best_features(X_train, y_train, y_test, features):
    '''
    function that returns a trained catboost model with the number of best featured calculated in jupyter notebook
    '''

    cbc = CatBoostRegressor(verbose=False)
    best_f = get_best_features(cbc.feature_importances_, X_train[features].columns, 'CB', cf.NUM_FEATURES)
    cbc = CatBoostRegressor(verbose=False)
    cbc.fit(X_train[best_f], y_train)
    return cbc

def main():

    calendar = pd.read_csv(cf.CALENDAR_FILE)
    listings = pd.read_csv(cf.LISTINGS_FILE)

    listings = preprocessing_listings(listings)
    listings_train,listings_test = final_preprocessing_listings(listings)

    calendar = preprocessing_calendar(calendar)

    listings_train,listings_test = adjustments(calendar,listings_train,listings_test)

    calendar = calendar_creation(calendar)

    listings_train,listings_test = data_creation(calendar,listings_train,listings_test)

    listings_train,listings_test = feature_creation(listings_train,listings_test)

    listings_train,listings_test = final_adjustments(listings_train,listings_test)

    X_train,y_train,X_test,y_test = split_Xy(listings_train,listings_test)
    new_features = cf.AMENITIES_LIST + cf.NEW_FEATURES
    old_features = list(set(X_train.columns) - set(new_features))

    X_train = X_train[old_features+new_features]
    X_test = X_test[old_features+new_features]

    cb_model = get_model_with_best_features(X_train,y_train,y_test,old_features+new_features)

    y_preds = cb_model.predict(X_test)
    r2 = r2_score(y_test,y_preds)


if __name__ == "__main__":
    main()
