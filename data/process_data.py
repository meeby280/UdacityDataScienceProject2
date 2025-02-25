import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    # Loading the messages file
    messages = pd.read_csv(messages_filepath)

    # Loading the categories file
    categories = pd.read_csv(categories_filepath)

    # Merging the two datasets based on the id and returning the dataframe
    return pd.merge(messages, categories, on="id", how="inner")


def clean_data(df):
    # Getting the categories column and splitting it into the separate categories
    categories = df["categories"].str.split(";", expand=True)
    # Getting the first row to get the individual categories
    row = categories.iloc[0]
    # Getting the category names by removing the last two characters and converting to a list
    category_colnames = row.apply(lambda x: x[:-2]).tolist()
    # Renaming the categories by the category names in the above list
    categories.columns = category_colnames

    # Iterating through the category columns to get the values for that column as an integer
    for column in categories:
        # The value is the last character
        categories[column] = categories[column].str[-1]
        # Making sure the character is saved as an integer
        categories[column] = categories[column].astype(int)

        # Making sure the values are either 0 or 1
        categories.loc[categories[column] > 1, column] = 1
        categories.loc[categories[column] < 0, column] = 0

    # Dropping the original categories column from the dataframe
    df.drop("categories", axis=1, inplace=True)

    # Joining the split categories columns back to the dataframe
    df = pd.concat([df, categories], axis=1)

    # Making sure there are no duplicate rows
    df.drop_duplicates(inplace=True)

    return df


# Creates a sqlite database and saves the dataframe to a table called disaster_messages
def save_data(df, database_filename):
    engine = create_engine(database_filename)
    df.to_sql("disaster_messages", engine, index=False, if_exists="replace")


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print(
            "Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}".format(
                messages_filepath, categories_filepath
            )
        )
        df = load_data(messages_filepath, categories_filepath)

        print("Cleaning data...")
        df = clean_data(df)

        print("Saving data...\n    DATABASE: {}".format(database_filepath))
        save_data(df, database_filepath)

        print("Cleaned data saved to database!")

    else:
        print(
            "Please provide the filepaths of the messages and categories "
            "datasets as the first and second argument respectively, as "
            "well as the filepath of the database to save the cleaned data "
            "to as the third argument. \n\nExample: python process_data.py "
            "disaster_messages.csv disaster_categories.csv "
            "DisasterResponse.db"
        )


if __name__ == "__main__":
    main()
