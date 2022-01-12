TO_DELETE = ['name','transit','street','neighbourhood_cleansed','picture_url','neighborhood_overview', 'space','host_about',
             'thumbnail_url','host_picture_url','description','medium_url','host_thumbnail_url','xl_picture_url','notes',
             'summary','neighbourhood_group_cleansed','host_url','listing_url','first_review',
             'calendar_last_scraped','last_review','host_name','calendar_updated','state','host_id','host_location',
             'scrape_id','square_feet','weekly_price','monthly_price','license','experiences_offered','city','market',
             'smart_location','country_code','country','requires_license','jurisdiction_names', 'security_deposit',
             'cleaning_fee','has_availability','host_acceptance_rate','host_total_listings_count']

TO_DELETE_FINAL = ['host_verifications','host_since','host_neighbourhood','amenities','id','last_scraped',
            'zipcode2','code']

TO_BOOLEAN = ['host_is_superhost','host_has_profile_pic','host_identity_verified','is_location_exact',
             'instant_bookable','require_guest_profile_picture','require_guest_phone_verification']

FROM_PERCENTAGE = ['host_response_rate']

TO_FILL_NA = ['host_neighbourhood','neighbourhood']

FILL_WITH_MEAN = ['host_response_rate','review_scores_rating','review_scores_accuracy',
                 'review_scores_cleanliness','review_scores_checkin','review_scores_communication',
                 'review_scores_location','review_scores_value','reviews_per_month']

FILL_WITH_MOSTF = ['host_response_time','bathrooms','bedrooms','beds','property_type']

AMENITIES_LIST = ['has_ac','has_internet','has_pool','has_tv','smoking_allowed','washer','smoke_detector',
                 'pet_allowed','kitchen','iron','fireplace','hot_tub','heating','gym','parking',
                 'elevator','doorman','breakfast']

AGGREGATE = ['neighbourhood','zipcode']

HOST_RESP_DICT  = {'a few days or more':1, 'within a day':2, 'within a few hours':3, 'within an hour':4}

CANCELATION_DICT = {'flexible':1, 'moderate':2, 'strict':3}

NEW_FEATURES = ['occupancy', 'weekend', 'summer', 'num_conf' ,'years_host' ,  'same_neighbourhood',
                                 'is_downtown' ,'distance_to_downtown', 'avg_rating_per_zipcode' ,
                                 'avg_prices_per_zipcode', 'avg_rating_radius', 'avg_prices_radius', 'restaurants_radius']

NUM_FEATURES = 74

CALENDAR_FILE = 'calendar.csv'
LISTINGS_FILE = 'listings.csv'
MIN_LAT = 47.60
MIN_LON = -122.40
MAX_LAT = 47.66
MAX_LON = -122.275
DOWNTOWN_LAT = 47.6254
DOWNTOWN_LON = -122.3355

API_KEY = 'B6FdYh1q4CLceKS84XrCQMxindMSIThCTT1jWphGoMbMrJAneCxV4D-jrmbxvdwBPrXhbUMHVWcdfC0YtiwaDb1pMEnJn8I4JUmF7w7i9NitlSu0FM4i1OUJRyO7YXYx'
URL = 'https://api.yelp.com/v3/businesses/search'
CREATE_YELP = False
RADIUS_YELP = False
YELPDATA_FILE = "yelpdata.pkl"