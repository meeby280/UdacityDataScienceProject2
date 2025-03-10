import sys

import nltk

nltk.download(["punkt", "wordnet"])

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import pandas as pd
import numpy as np

from sqlalchemy import create_engine

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import make_scorer, accuracy_score, f1_score, precision_score, recall_score

import re

import pickle

# A dictionary of scorers to use in the GridSearchCV evaluation
evaluation_scorers = {
    'accuracy': make_scorer(  accuracy_score ),
    'f1': make_scorer(  f1_score, zero_division=0, average='weighted' ),
    'precision': make_scorer(  precision_score, zero_division=0, average='weighted' ),
    'recall': make_scorer(  recall_score, zero_division=0, average='weighted' )
}


def load_data(database_filepath):
    """
    Loads the data from the database and splits into the message values and the categories.

    Parameters:
    database_filepath (str): The path to the sqlite database file.

    Returns:
    X: The values for the messages column of the 'disaster_messages' table.
    Y: The values for the categorical columns of the 'disaster_messages' table.
    category_names: The names of the categorical columns.
    """
    # Create the engine
    engine = create_engine(f"sqlite:///{database_filepath}")
    # Load the data in a pandas Dataframe
    df = pd.read_sql_table("disaster_messages", engine)

    # Gets the values for the 'message' column
    X = df.message.values
    # These are the categorical columns in the dataset. Excluding the categories that have only one value.
    category_names = [cn for cn in df.columns[4:] if df[cn].nunique() > 1]
    # Our Y should be only the part of the dataframe that contains the categories and the values for each record.
    Y = df[category_names].values

    return X, Y, category_names


def tokenize(text):
    """
    Tokenizes a string of text. Changes the case to lower case and removes any non-alphanumeric characters.

    Parameters:
    text (str): The string of text to clean and separate into tokens.

    Returns:
    list: A list of cleaned tokens for the input text.
    """
    # Remove any non-alphanumeric characters and change the case to lower case
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    # Tokenize the text using NLTK's WordNetLemmatizer
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Builds the model using a pipeline and passing it to GridSearchCV. Refits the model according to the best scored parameters.

    Parameters:
    None

    Returns:
    GridSearchCV: The model to use for the classification.
    """
    # The creating a model that uses HashingVectorizer to vectorize the text and LinearSVC as the classifier.
    base_model = Pipeline(
        [
            ("vect", HashingVectorizer(tokenizer=tokenize, token_pattern=None)),
            ("clf", MultiOutputClassifier(LinearSVC(dual="auto"))),
        ]
    )

    # Parameter set up for GridSearchCV
    parameters = {
        "vect__ngram_range": ((1, 1), (1, 2)),
        "clf__estimator__C": [0.1, 1, 10],
        "clf__estimator__max_iter": [1000, 2000, 3000],
        "clf__estimator__penalty": ["l1", "l2"],
    }

    # Create the model using GridSearchCV so that we can find the best parameters for the model.
    model = GridSearchCV(base_model, param_grid=parameters, n_jobs=-1, refit='accuracy', scoring=evaluation_scorers, return_train_score=True)
    return model


def evaluate_model(model, X_test):
    """
    Calculates the accuracy, precision, and recall for the model based on the test data.

    Parameters:
    model: The model to evaluate.
    X_test: The test data to evaluate the model on.

    Returns:
    None
    """
    # Predict the values for the test data
    model.predict(X_test)

    # Get the results from the GridSearchCV evaluation
    results = model.cv_results_
    best_params = model.best_params_

    # The target scorer is the first scorer in the evaluation_scorers dictionary
    target_scorer = list(evaluation_scorers.keys())[0]
    target_scorer_idx = model.best_index_

    # Print the scores for the first scorer in the evaluation_scorers dictionary and what the parameters were for this best score
    print(f"\tBest Parameters: {best_params}\n\tScores for best '{target_scorer}' score:")
    for scorer in list(evaluation_scorers.keys()):
        curr_score = results[f"mean_test_{scorer}"][target_scorer_idx] * 100
        print(f"\t {scorer.ljust(10, ' ')}: {curr_score:.2f}%")


def save_model(model, model_filepath):
    """
    Saves the model to the specified filepath.

    Parameters:
    model (obj): The model to save.
    model_filepath (str): The path to save the model to.
    """
    pickle.dump(model, open(model_filepath, "wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print("Loading data...\n    DATABASE: {}".format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print("Building model...")
        model = build_model()

        print("Training model...")
        model.fit(X_train, Y_train)

        print("Evaluating model...")
        evaluate_model(model, X_test)

        print("Saving model...\n    MODEL: {}".format(model_filepath))
        save_model(model, model_filepath)

        print("Trained model saved!")

    else:
        print(
            "Please provide the filepath of the disaster messages database "
            "as the first argument and the filepath of the pickle file to "
            "save the model to as the second argument. \n\nExample: python "
            "train_classifier.py ../data/DisasterResponse.db classifier.pkl"
        )


if __name__ == "__main__":
    main()
