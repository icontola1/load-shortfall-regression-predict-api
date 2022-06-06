"""
    Simple file to create a sklearn model for deployment in our API

    Author: Explore Data Science Academy

    Description: This script is responsible for training a simple linear
    regression model which is used within the API for initial demonstration
    purposes.

"""

# Dependencies
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import json

# Fetch training data and preprocess for modeling
train = pd.read_csv('./data/df_train.csv')
                                  
# Splitting our data into dependent Variable and Independent Variable
#y_train = train['load_shortfall_3h'].astype('int')

y_train =  train[['load_shortfall_3h']]

x_train = train[['Madrid_wind_speed', 'Bilbao_rain_1h',
   'Valencia_wind_speed', 'Seville_humidity', 'Madrid_humidity',
   'Bilbao_clouds_all', 'Bilbao_wind_speed', 'Seville_clouds_all',
   'Bilbao_wind_deg', 'Barcelona_wind_speed', 'Barcelona_wind_deg',
   'Madrid_clouds_all', 'Seville_wind_speed', 'Barcelona_rain_1h', 'Seville_rain_1h', 'Bilbao_snow_3h',
   'Barcelona_pressure', 'Seville_rain_3h', 'Madrid_rain_1h',
   'Barcelona_rain_3h', 'Valencia_snow_3h', 'Madrid_weather_id',
   'Barcelona_weather_id', 'Bilbao_pressure', 'Seville_weather_id',
   'Seville_temp_max', 'Madrid_pressure',
   'Valencia_temp_max', 'Valencia_temp', 'Bilbao_weather_id',
   'Seville_temp', 'Valencia_humidity', 'Valencia_temp_min',
   'Barcelona_temp_max', 'Madrid_temp_max', 'Barcelona_temp',
   'Bilbao_temp_min', 'Bilbao_temp', 'Barcelona_temp_min',
   'Bilbao_temp_max', 'Seville_temp_min', 'Madrid_temp', 'Madrid_temp_min']]
x_train = np.array(x_train)
    
# Fitting the model
ran_forest = RandomForestRegressor(n_estimators=200, max_depth=8)
print ("Training Model...")
ran_forest.fit(x_train,y_train)

# Pickle model for use within our API
save_path = '../assets/trained-models/randforest.pkl'
print (f"Training completed. Saving model to: {save_path}")
pickle.dump(ran_forest, open(save_path,'wb'))


