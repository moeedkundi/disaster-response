import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import os
import pickle
import nltk
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score, fbeta_score, classification_report
from scipy.stats import hmean
from scipy.stats.mstats import gmean
import re
from sklearn.preprocessing import FunctionTransformer
from sklearn.externals import joblib

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])





def load_data(data_file):
    # read in file
    engine = create_engine('sqlite:///' + data_file)
    df = pd.read_sql_table('category-data', engine)

    #Input and Targets
    X = df.message.values
    y = df.iloc[:,4:]
    return X, y

def tokenize(text):
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlink")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    lem_tokens = []
    for t in tokens:
        clean_t = lemmatizer.lemmatize(t).lower().strip()
        lem_tokens.append(clean_t)

    return lem_tokens



class get_len(BaseEstimator, TransformerMixin):
    # extractor/transformer to find all uppercase words
    def transform(self, X, y=None):
        X_transformed =np.array([len(text) for text in X]).reshape(-1, 1)
        return X_transformed

    def fit(self, X, y=None):
        return self

def build_model():
    pipeline = Pipeline(
        [('features', FeatureUnion([
            ('pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
            ('length', Pipeline([
            ('length', get_len()),
        ]))
        ])),
         

         ('clf', MultiOutputClassifier(RandomForestClassifier()))
         ])

    parameters = {'clf__estimator__min_samples_split': [2, 3]}
    model = GridSearchCV(pipeline, param_grid=parameters, n_jobs=5)

    return model


def train(X, y, model):

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    model = []
    model = build_model()

    print('Training Model: ')

    model.fit(X_train, y_train)

    print('Now Evaluating Trained model: ')

    labels=y.columns.tolist()

    test_model(model,X_test,y_test,labels)


    return model

def test_model(model, X_test, y_test, labels):
    #Evaluates trained model

    preds = model.predict(X_test)

    for label in range(0, len(labels)):
        print('Message category:', labels[label])
        print(classification_report(y_test.iloc[:, label], preds[:, label]))


def export_model(model,output_file):
    # Export model as a pickle file
    joblib.dump(model, open(output_file, 'wb'))



def run_pipeline(data_file):
    X, y = load_data(data_file)  # run ETL pipeline
    model = build_model()  # build model pipeline
    model = train(X, y, model)  # train model pipeline
    export_model(model,output_file)  # save model


if __name__ == '__main__':
    data_file = sys.argv[1]  # get filename of dataset
    output_file = sys.argv[2]
    run_pipeline(data_file)  # run data pipeline
