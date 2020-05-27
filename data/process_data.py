"""
Cleaning and Preprocessing Data

Script Execution Example:
> python process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
Arguments:
    1) disaster_messages.csv => CSV containing messages
    2) disaster_categories.csv => CSV containing categories
    3) DisasterResponse.db => SQLite database name which is to be stored as output
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Data Loading function
    
    Arguments:
        messages_filepath => path to messages csv file
        categories_filepath => path to categories csv file
    Output:
        df => Loaded DataFrame
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories,on='id',how='inner')
    return df
    


def clean_data(df):
    """
    Data Cleaning function
    
    Arguments:
        df => Uncleaned DataFrame
    Outputs:
        df => Cleaned DataFrame
    """
    categories = df['categories'].str.split(';',expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x.split('-')[0])
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].apply(lambda x: int(x.split('-')[1]))
    df=df.drop('categories',axis=1)
    df = pd.concat([df,categories],axis=1)
    df=df.drop_duplicates()
    
    return df


def save_data(df, database_filename):
    """
    Save Data
    
    Arguments:
        df => Clean DataFrame
        database_filename -> Database destination
    """
#     engine = create_engine(database_filename)
    df.to_sql('disaster_category_data', 'sqlite:///'+database_filename, index=False)  


def main():
    """
    Main function executes:
        1) Data Loading
        2) Data Cleaning
        3) Data Saving
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
