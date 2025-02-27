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
from sklearn.metrics import classification_report

import re

import pickle


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
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Builds the model using a pipeline and passing it to GridSearchCV. Refits the model according to the best scored parameters.

    Parameters:
    text (str): The string of text to clean and separate into tokens.

    Returns:
    list: A list of cleaned tokens for the input text.
    """
    base_model = Pipeline(
        [
            ("vect", HashingVectorizer(tokenizer=tokenize, token_pattern=None)),
            ("clf", MultiOutputClassifier(LinearSVC(dual="auto"))),
        ]
    )

    parameters = {
        "vect__ngram_range": ((1, 1), (1, 2)),
        "clf__estimator__C": [0.1, 1, 10],
        "clf__estimator__max_iter": [1000, 2000, 3000],
        "clf__estimator__penalty": ["l1", "l2"],
    }

    model = GridSearchCV(base_model, param_grid=parameters, n_jobs=-1, refit=True)
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Calculates the accuracy, precision, and recall for the model based on the test data.

    Parameters:
    model: The model to evaluate.
    X_test: The test data to evaluate the model on.
    Y_test: The true values for the test data.
    category_names: The names of the categories in the dataset.

    Returns:
    list: A list of cleaned tokens for the input text.
    """
    Y_pred = model.predict(X_test)

    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    max_len = max([len(cn) for cn in category_names])

    for col_idx, cn in enumerate(category_names):
        report = classification_report(
            Y_test[:, col_idx],
            Y_pred[:, col_idx],
            zero_division=np.nan,
            output_dict=True,
        )
        col_acc = report["accuracy"]
        col_prec = report["macro avg"]["precision"]
        col_recl = report["macro avg"]["recall"]
        col_f1 = report["macro avg"]["f1-score"]

        accuracy_scores.append(col_acc)
        precision_scores.append(col_prec)
        recall_scores.append(col_recl)
        f1_scores.append(col_f1)

        print(
            f"\t{cn.rjust(max_len, ' ')}: Accuracy: {col_acc:.2f} Precision: {col_prec:.2f} Recall: {col_recl:.2f} F1 Score: {col_f1:.2f}"
        )

    print("\t--- Overall Averaged Scores ---")
    print(f"\tAccuracy: {np.mean(accuracy_scores):.2f}")
    print(f"\tPrecision: {np.mean(precision_scores):.2f}")
    print(f"\tRecall: {np.mean(recall_scores):.2f}")
    print(f"\tF1: {np.mean(f1_scores):.2f}")
    print()

    return model


def save_model(model, model_filepath):
    """
    Builds the model using a pipeline.

    Parameters:
    text (str): The string of text to clean and separate into tokens.

    Returns:
    list: A list of cleaned tokens for the input text.
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
        evaluate_model(model, X_test, Y_test, category_names)

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
