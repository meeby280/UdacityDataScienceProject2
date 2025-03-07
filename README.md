# Disaster Response Pipeline Project

## Summary
### Data Loading (ETL) `data/process_data.py`
- Loads data from CSV files for disaster messages and categories (`data/disaster_messages.csv` and `data/disaster_categories.csv`).
- Cleans the categories data to create separate columns for each category.
- Separates the value for each category column so that a value of 0 or 1 is applied for whether the message for that row falls into that category.
- Merges the message data and category data to provide a full view of the message and its category classifications.
- Stores the merged data into a sqlite database table named `disaster_messages` within the `data/DisasterResponse.db` database.
### Model Training `models/train_classifier.py`
- Loads the data from the table named `disaster_messages` within the `data/DisasterResponse.db` database.
- Tokenizes the messages and applies lemmatization (`nltk` library, `WordNetLemmatizer`) to get the base word for each token.
- Builds and trains a `scikit-learn` model using `HashingVectorizer` for token vectorization and `LinearSVC` within a `MultiOutputClassifier` for classification.
- The model is evaluated based on `GridSearchCV`'s results for model accuracy. Parameters currently used for tuning are:
    - ngram_range: Vectorizer.
        - Testing values: ((1, 1), (1, 2))
    - C: Classifier.
        - Testing values: [0.1, 1, 10]
    - max_iter: Classifier.
        - Testing values: [1000, 2000, 3000]
    - penalty: Classifier.
        - Testing values: [l1, l2]
- The trained model is saved as `models/classifier.pkl`.
### Web App `app/run.py`
- Flask-based web app using `plotly` for visualizing data.
- Contains 3 graphs based on the data in the `disaster_messages` table created in the [Data Loading](#data-loading-etl-dataprocess_datapy) step.
    - The first graph is a bar chart with the counts of messages per each category for the top ten categories.
    - The second graph is a pie chart for counts of message word counts. First the counts of words per message are calculated. Then the messages are grouped into bins of 10 to see how many messages fall into each bin.
        - Note: There were not many messages above 50 words, so these are grouped into one bin (50+) even though there were messages with word counts in the thousands.
    - The third graph is a bar chart that shows how many messages per category fall into each of the bins. Using what was calculated before for the second graph, the percentage of each bin within each category is calculated.
- Additionally, the user can choose to classify a message using the model trained in the [Model Training](#model-training-modelstrain_classifierpy) step. This will try to classify the the user's inputted message according to one of the 36 categories.

## Instructions for running:
1. Run the following commands in the project's root directory to set up the database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run the web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/