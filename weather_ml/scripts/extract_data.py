import pyodbc
import pandas as pd

def extract_data(connection):
    
    try:
        query = "SELECT * FROM Weather_Traffic_Data"

        cursor = connection.cursor()
        cursor.execute(query)

        # Get column names
        columns = [column[0] for column in cursor.description]

        # Fetch data as dictionaries
        data = [dict(zip(columns, row)) for row in cursor.fetchall()]

        # Convert to DataFrame
        raw_data = pd.DataFrame(data)

        cursor.close()
        connection.close()

        return raw_data
    except Exception as e:
        #logging.error(f"Error occurred during data extraction: {e}")
        print(f"An error occurred during data extraction: {e}")
        return None