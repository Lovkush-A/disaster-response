# The machine learning script, train_classifier.py, runs in the terminal without errors. The script takes the database file path and model file path, creates and trains a classifier, and stores the classifier into a pickle file to the specified model file path.

# The script uses a custom tokenize function using nltk to case normalize, lemmatize, and tokenize text. This function is used in the machine learning pipeline to vectorize and then apply TF-IDF to the text.

# The script builds a pipeline that processes text and then performs multi-output classification on the 36 categories in the dataset. GridSearchCV is used to find the best parameters for the model.

# The TF-IDF pipeline is only trained with the training data. The f1 score, precision and recall for the test set is outputted for each category.

import sys
import pandas as pd
from sqlalchemy import create_engine

# packages for nlp
import re
import nltk
nltk.download('punkt') # needed for word_tokenize
nltk.download('wordnet') # needed for wordnetlemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# imports for ml pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# pickling
from joblib import dump, load

def load_data(database_filepath):
    """
    load dataset and return X dataframe, Y dataframe and
    category names as list of string
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('disaster', engine)

    X = df.message
    Y = df.drop(columns=['id', 'message', 'original', 'genre'])
    category_names = Y.columns

    return X, Y, category_names

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


def build_model():
    pipeline = Pipeline(steps=[
        ('vectorize', TfidfVectorizer(tokenizer = tokenize, stop_words='english')),
        ('clf', MultiOutputClassifier(RandomForestClassifier(random_state=0)))
    ])
    
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)

    for i, column in enumerate(Y_test.columns):
        print(column)
        print(classification_report(Y_test.iloc[:,i], Y_pred[:,i]))
        print('\n\n'+'='*10)
    
    return None


def save_model(model, model_filepath):
    dump(model, model_filepath)
    return None


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()