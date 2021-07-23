# The web app, run.py, runs in the terminal without errors. The main page includes at least two visualizations using data from the SQLite database.

# When a user inputs a message into the app, the app returns classification results for all 36 categories.

import json
import plotly
import pandas as pd

import re
import nltk
nltk.download('punkt') # needed for word_tokenize
nltk.download('wordnet') # needed for wordnetlemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sqlalchemy import create_engine

from joblib import load


app = Flask(__name__)

# def tokenize(text):
#     tokens = word_tokenize(text)
#     lemmatizer = WordNetLemmatizer()

#     clean_tokens = []
#     for tok in tokens:
#         clean_tok = lemmatizer.lemmatize(tok).lower().strip()
#         clean_tokens.append(clean_tok)

#     return clean_tokens

def tokenize(text: str):
    """
    replace urls, lemmatize and tokenize
    """
    # replace urls with 'urlplaceholder'
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    text = re.sub(pattern=url_regex, repl='urlplaceholder', string=text)

    lemmatizer = WordNetLemmatizer()
    
    tokens = word_tokenize(text)
    clean_tokens = [lemmatizer.lemmatize(token).strip() for token in tokens]

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/disaster.db')
df = pd.read_sql_table('disaster', engine)

# load model
model = load("../models/classifier.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for udacity example
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # extract data for category plot
    category_percentages = (
        df
        .drop(columns=['id', 'message', 'original', 'genre'])
        .mean()
        .sort_values(ascending=False)
    )
    category_names = category_percentages.index
    
    # extract data for word frequencies
    tokenized_messages = [tokenize(text) for text in df.message]
    
    word_count = {}
    for message in tokenized_messages:
        for token in message:
            if token not in word_count:
                word_count[token] = 0
            word_count[token] += 1
    
    word_count_sorted = sorted(word_count.items(), key=lambda item: item[1], reverse=True)
    
    top_words = [word for word,_ in word_count_sorted[0:50]]
    top_words_count = [count for _,count in word_count_sorted[0:50]]
    
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_percentages
                )
            ],

            'layout': {
                'title': 'Percentage of total messages in each category',
                'yaxis': {
                    'title': "% of all messages in that category"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=top_words,
                    y=top_words_count
                )
            ],

            'layout': {
                'title': 'Most frequent "words" in the dataset',
                'yaxis': {
                    'title': "Number of occurences of the word"
                },
                'xaxis': {
                    'title': "Word"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
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