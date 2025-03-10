import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie
import plotly.graph_objs as go
import plotly.express as px

import joblib
from sqlalchemy import create_engine

import re

app = Flask(__name__)


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

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disaster_messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # Counts per category for the top 10 categories
    # Gets the columns for the message categories
    msg_category_cols = df.columns[4:]
    # Counts of messages per category and gets the top 10 categories by count.
    cat_count = df[ msg_category_cols ] \
                .sum() \
                .sort_values( ascending = False ) \
                .head( 10 ) \
                .reset_index() \
                .rename( columns = { 'index': 'category', 0: 'count' } )

    # Count of messages per length bin
    # Creating the bins and labels for the word count bins
    bins = [ 0, 10, 20, 30, 40, 50, float('inf') ]
    labels = [ '0-9', '10-19', '20-29', '30-39', '40-49', '50+' ]
    # Calculates the word count per each message using the tokenize function
    word_count = df['message'].apply( lambda msg: len( tokenize(msg) ) ).rename('word_count').reset_index(drop=True)
    # Uses the pandas cut method to bin the word counts according to those created above
    df['word_count_bin'] = pd.cut( word_count, bins=bins, labels=labels )
    # Counts the number of messages that fall into each bin
    count_per_bin = df.groupby(['word_count_bin'], observed=False).agg( count=('word_count_bin', 'count') ).reset_index()

    # Counts of word count bins per categories
    # Gets the counts per bin and category
    cat_wordcount_counts = df[['word_count_bin', *msg_category_cols]].groupby(['word_count_bin'], observed=False).sum().reset_index()
    # Restructures the dataframe so that the category columns become one 'category' column
    count_by_category_and_bin = pd.melt( cat_wordcount_counts, id_vars=['word_count_bin'], value_vars=msg_category_cols, var_name='category', value_name='count' )
    # Gets the counts per category so that we can calculate the percentage later
    total_counts = count_by_category_and_bin.groupby('category', observed=False)['count'].sum().reset_index().rename(columns={'count': 'total_count'})
    # Merges the total counts per category back into the main dataframe
    count_by_category_and_bin = count_by_category_and_bin.merge(total_counts, on='category')
    # Calculates the percentage of messages per category and bin
    count_by_category_and_bin['percentage'] = count_by_category_and_bin['count'] / count_by_category_and_bin['total_count'] * 100
    # Rounds the percentage to 2 decimal places for better display
    count_by_category_and_bin['percentage'] = count_by_category_and_bin['percentage'].apply( lambda x: round(x, 2) )
    # Sorts the dataframe by category and word count bin
    count_by_category_and_bin.sort_values(by=['category', 'word_count_bin'], ascending=[True, False], inplace=True)

    # This is where the graph objects are stored into as dictionary objects
    graphs = []

    # Graph 1: Count of messages per category
    cat_counts_graph = px.bar(
        data_frame=cat_count,
        x='category',
        y='count',
        title='Count of Messages by Category',
        labels={
            'x': 'Message Category',
            'y': 'Count'
        }
    )
    graphs.append( cat_counts_graph.to_dict() )

    # Graph 2: Count of messages per word count bin
    word_count_graph = px.pie(
        data_frame=count_per_bin,
        values='count', 
        names='word_count_bin', 
        title='Counts of Message Lengths',
        labels={
            'word_count_bin': 'Word Count',
            'count': 'Count'
        }
    )
    graphs.append( word_count_graph.to_dict() )

    # Graph 3: Count of messages per category and word count bin
    percentage_per_bin_graph = px.bar(
        data_frame=count_by_category_and_bin, 
        x='category', 
        y='percentage', 
        title='Percentage of Messages per Category and Word Count Bin',
        labels={
            'x': 'category',
            'y': 'percentage'
        },
        color='word_count_bin', 
        barmode='stack'
    )
    graphs.append( percentage_per_bin_graph.to_dict() )
    
    # encode plotly graphs in JSON for rendering in Javascript
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()