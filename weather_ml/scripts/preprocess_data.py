import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def clean_data(data):
    
    data = data.drop(columns=["ID", "timestamp"])
    
    data.dropna(inplace=True)
    
    data.drop_duplicates(inplace=True)
    
    data['extraction_time'] = pd.to_datetime(data['extraction_time'])
    data['hour'] = data['extraction_time'].dt.hour
    data['day'] = data['extraction_time'].dt.day
    data['month'] = data['extraction_time'].dt.month
    data['weekday'] = data['extraction_time'].dt.day_name()
    
    return data

def prepare_data(data):
    
    data.sort_values(by='extraction_time', inplace=True)
    label_encoder = LabelEncoder()
    data['weather_encoded'] = label_encoder.fit_transform(data['weather'])
    data['weekday_encoded'] = label_encoder.fit_transform(data['weekday'])
    
    data = data.drop(['city', 'weather', 'weekday'], axis=1)
    
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(data.drop(['extraction_time'], axis=1))
    
    data_ml = pd.DataFrame(scaled_features, columns=data.columns[1:])
    
    return data_ml