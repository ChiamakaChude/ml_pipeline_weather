import pyodbc
import pandas as pd

from config.config import username, password

def connect_db(server, database, driver):
    
    try:
        #logging.info(f"Connection to database '{database}'...")
        #Initial connection to the server
        connection_string = f'DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password};Trusted_Connection=no;'
        connection = pyodbc.connect(connection_string)
        cursor = connection.cursor()

        # Check if the database exists
        cursor.execute(f"SELECT name FROM sys.databases WHERE name = ?", database)
        db_exists = cursor.fetchone()

        if not db_exists:
            #If database does not exist, create it
            #logging.warning(f"Database '{database}' does nto exist. Creating it...")
            
            cursor.execute(f"CREATE DATABASE {database}") #execute create query

            #logging.info(f"Database '{database}' created successfully.")
            
        else:
            #logging.info(f"Connection to database '{database}' successful!!!")
            print("connection succesful")
            
        return connection #return the database connection
        
    except Exception as e:
        #If there is an error
        print(f"Unexpected error occured {e}")
        #logging.error(f"Error occurred during database connection: {e}")
        return None
