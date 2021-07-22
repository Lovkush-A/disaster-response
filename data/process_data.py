import sys

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

# The ETL script, process_data.py, runs in the terminal without errors. The script takes the file paths of the two datasets and database, cleans the datasets, and stores the clean data into a SQLite database in the specified database file path.


def load_data(
    messages_filepath: str,
    categories_filepath: str
) -> pd.DataFrame:
    """
    load csv files, call expand categories, merge using 'id' column,
    and return dataframe
    """
    messages = pd.read_csv(messages_filepath)
    
    categories = pd.read_csv(categories_filepath)
    categories = expand_categories(categories)    
    
    df = pd.merge(
        left=messages,
        right=categories,
        how='outer',
        on='id'
    )

    return df


def expand_categories(categories: pd.DataFrame) -> pd.DataFrame:
    """
    split and expand strings in categories dataframe, create column names, and change
    entries into 1's and 0's
    """
    categories_expanded = categories.categories.str.split(pat=';', expand=True)
    
    # column names taken from first row, as each entry is
    # of the form 'category-0' or 'category-1'
    columns = [category[:-2] for category in categories_expanded.iloc[0]]
    categories_expanded.columns = columns
    
    # replace each entry with the 0 or 1 at the end, then convert to int32 type
    for column in columns:
        categories_expanded[column] = ( 
            categories_expanded[column]
            .astype(str)
            .str.get(-1)
            .astype('int32')
        )
    
    # add the primary key to new frame from old frame
    categories_expanded['id'] = categories.id
    
    return categories_expanded


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    remove duplicate entries
    """
    # not providing subset resulted in some duplicates remaining
    # due to time restrictions, the exact cause of this is not known
    # but would be investigated if more time was available
    df = df.drop_duplicates(subset = ['id', 'message'])
    return df


def save_data(df, database_filename):
    """
    save dataframe in sql database
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('disaster', engine, index=False)
    pass  


def main():
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