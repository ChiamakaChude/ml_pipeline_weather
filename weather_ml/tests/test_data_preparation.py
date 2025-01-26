import unittest
import pandas as pd

from scripts.model import modelling
from scripts.preprocess_data import clean_data, prepare_data
from scripts.database_connection import connect_db
from scripts.extract_data import extract_data
from config.config import server, database, driver


from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout

from tensorflow.keras.models import Sequential


connection = connect_db(server, database, driver)
raw_data = extract_data(connection)

data = raw_data.head(5)

class TestCleanData(unittest.TestCase):
    def test_clean_data(self):
        

        sample_data = data.head(5)
        sample_data.loc[2, "feels_like"] = None  #Introduce a missing value (edge case)

        # Convert extraction_time in expected_data to datetime
        sample_data["extraction_time"] = pd.to_datetime(sample_data["extraction_time"])
        
        # Call the function
        cleaned_data = clean_data(sample_data)
        
        prepared_data = prepare_data(cleaned_data)

        # Check the output
        self.assertTrue("weather_encoded" in prepared_data.columns)
        self.assertTrue("weekday_encoded" in prepared_data.columns)

        # Assert the output matches the expected data
        #pd.testing.assert_frame_equal(output_data.reset_index(drop=True), sample_data)
        return prepared_data


class TestModelling(TestCleanData):

    def test_modelling(self):

        prepared_data = self.test_clean_data()

        # Call the modelling function
        model, history = modelling(prepared_data, save_model=False)

        # Verify model properties
        self.assertEqual(len(model.layers), 3)  # Check number of layers
        self.assertIsInstance(model.layers[0], LSTM)  # Check first layer is LSTM
        self.assertIsInstance(model.layers[1], Dropout)  #Check second layer is Dropout
        self.assertIsInstance(model.layers[2], Dense)  #Check third layer is Dense

        # Verify training history
        self.assertIn('loss', history.history)  # Check if loss is in history
        self.assertIn('val_loss', history.history)  # Check if validation loss is in history
        self.assertGreater(len(history.history['loss']), 0)  # Check if epochs were run


if __name__ == "__main__":
    #unittest.main()
    unittest.TextTestRunner().run(unittest.TestLoader().loadTestsFromTestCase(TestCleanData))
