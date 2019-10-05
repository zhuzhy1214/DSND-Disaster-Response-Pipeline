import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """loads the message and category data
    Args:
        messages_filepath (string): The file path of the messages csv
        categories_filepath (string): The file path of the categories cv
    Returns:
        df (pandas dataframe): The combined messages and categories df
    """

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id', how='inner')

    return df


def clean_data(df):
    """Cleans the data:
        - drops duplicates
        - removes messages missing classes
        - cleans up the categories column
    Args:
        df (pandas dataframe): combined categories and messages df
    Returns:
        df (pandas dataframe): Cleaned dataframe with split categories
    """

    categories = df['categories'].str.split(';', expand=True)

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    #     #regulate some of the '2' value to '1'
    #     categories['related']  = categories['related'].apply(lambda x: min(x,1))

    df.drop(columns=['categories'], inplace=True)
    df = pd.concat([df, categories], axis=1)

    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename, table_name='processed_data'):
    """Saves the resulting data to a sqlite db
    Args:
        df (pandas dataframe): The cleaned dataframe
        database_filename (string): the file path to save the db
    Returns:
        None
    """

    engine = create_engine('sqlite:///' + database_filename)
    sql = 'DROP TABLE IF EXISTS {};'.format(table_name)
    result = engine.execute(sql)

    df.to_sql(table_name, engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath, 'processed_data')

        print('Cleaned data saved to database with table name of {}!'.format('processed_data'))

    else:
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()