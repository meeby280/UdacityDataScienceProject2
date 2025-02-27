import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie
import plotly.graph_objs as go
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
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
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # Counts per category for the top 10 categories
    msg_category_cols = df.columns[4:]
    cat_count = df[msg_category_cols].sum().sort_values(ascending=False).head(10)
    msg_category_display = [ cn.replace("_", " ").capitalize() for cn in cat_count.index ]

    # Count of messages per length bin
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, float('inf')]
    labels=['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100', '100+']
    word_count = df['message'].apply( lambda msg: len(tokenize(msg)) ).rename('word_count').reset_index(drop=True)
    word_count_bins = pd.cut( word_count, bins=bins, labels=labels )
    count_per_bin = word_count_bins.value_counts().sort_index().reset_index().rename(columns={'word_count': 'bins'})
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=msg_category_display,
                    y=cat_count
                )
            ],
            'layout': {
                'title': {
                    'text': 'Top 10 Categories of Messages by Count'
                },
                'yaxis': {
                    'title': {
                        'text': "Count"
                    }
                },
                'xaxis': {
                    'title': {
                        'text': "Message Category"
                    }
                }
            }
        }, {
            'data': [
                Pie(
                    labels=count_per_bin['bins'], 
                    values=count_per_bin['count'], 
                    sort=False
                )
            ],
            'layout': {
                'title': {
                    'text': 'Counts of Message Lengths'
                }
            }
        }, {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],
            'layout': {
                'title': {
                    'text': 'Distribution of Message Genres'
                },
                'yaxis': {
                    'title': {
                        'text': "Count"
                    }
                },
                'xaxis': {
                    'title': {
                        'text': "Genre"
                    }
                }
            }
        }, 
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    print(graphJSON)
    
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