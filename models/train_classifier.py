import sys
import pandas as pd
import numpy as np

#io
from sqlalchemy import create_engine
import pickle

#NLP
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


#preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split


#model
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import GridSearchCV

#metric
from sklearn.metrics import classification_report



def load_data(database_filepath, table_name = 'processed_data' ):
    """Loads X and Y and gets category names
    Args:
        database_filepath (str): string filepath of the sqlite database
    Returns:
        X (pandas dataframe): Feature data, just the messages
        Y (pandas dataframe): Classification labels
        category_names (list): List of the category names for classification
    """
    
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table(table_name , engine)
    
    X = df['message']
    y = df.drop(['message', 'genre', 'id', 'original'], axis=1)
    
    category_names = list(y.columns)
    return X, y, category_names

def tokenize(text):
    """Basic tokenizer that changes to lower case, removes punctuation and stopwords then lemmatizes
    Args:
        text (string): input message to tokenize
    Returns:
        clean_tokens (list): list of cleaned tokens in the message
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    """Returns the GridSearchCV object to be used as the model
    Args:
        None
    Returns:
        cv (scikit-learn GridSearchCV): Grid search model object
    """
    pipeline_message = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                             ('tfidf', TfidfTransformer())
                            ])

    model = RidgeClassifier()

    pipeline_2 = Pipeline([('message', pipeline_message), 
                         ('clf',model)
                        ])
    
    parameters = {
        # 'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        # 'features__text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
        # 'features__text_pipeline__vect__max_features': (None, 5000, 10000),
        # 'features__text_pipeline__tfidf__use_idf': (True, False),
        'clf__estimator__normalize': (True, False),
        'clf__estimator__alpha': (1.0, 0.8, 0.6)
    }
    cv = GridSearchCV(pipeline_2, param_grid=parameters)

    return cv
    

def evaluate_model(model, X_test, y_test, category_names):
    """Prints multi-output classification results
    Args:
        model (pandas dataframe): the scikit-learn fitted model
        X_text (pandas dataframe): The X test set
        Y_test (pandas dataframe): the Y test classifications
        category_names (list): the category names
    Returns:
        None
    """
    #TODO make sure category_names and y_test have the same dimension in labels
    y_pred_df = pd.DataFrame(model.predict(X_test), columns = category_names)
    y_test_df = pd.DataFrame(y_test, columns = category_names)
    for i in range(y_test.shape[1]):
        print('category name: {}'.format(category_names[i]))
        print(classification_report(y_test_df.iloc[:,i], y_pred_df.iloc[:,i]))

        # prediction, recall, f1, support = precision_recall_fscore_support(y_test[:, i].flatten(), y_pred[:, i].flatten())
        # print('prediction: {}'.format(prediction))
        # print('recall: {}'.format(recall))
        # print('f1: {}'.format(f1))
        # print('support: {}'.format(support))

def save_model(model, model_filepath):
    """dumps the model to the given filepath
    Args:
        model (scikit-learn model): The fitted model
        model_filepath (string): the filepath to save the model to
    Returns:
        None
    """    
    pickle.dump(model, open(model_filepath, "wb" ) )


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        cv = build_model()
        
        print('Training and optimizing model...')
        cv.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(cv, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(cv.best_estimator_, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()