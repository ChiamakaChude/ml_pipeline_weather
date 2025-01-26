import pyodbc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import unittest

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

from config.config import server, database, driver

from scripts.database_connection import connect_db
from scripts.extract_data import extract_data
from scripts.preprocess_data import clean_data
from scripts.preprocess_data import prepare_data
from scripts.model import modelling

from tests.test_data_preparation import TestCleanData
from tests.test_data_preparation import TestModelling

#test code
if __name__=="__main__":
    connection = connect_db(server, database, driver)
    data = extract_data(connection)
    
    #unittest.TextTestRunner().run(unittest.TestLoader().loadTestsFromTestCase(TestCleanData))
    unittest.TextTestRunner().run(unittest.TestLoader().loadTestsFromTestCase(TestModelling))

    #data_n = clean_data(data)
    #final_data = prepare_data(data_n, save_model=True)
    #model, history = modelling(final_data)