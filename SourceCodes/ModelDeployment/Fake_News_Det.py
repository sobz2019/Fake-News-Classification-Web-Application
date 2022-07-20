import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import json
import tldextract   # Accurately separates a URL's subdomain, domain, and public suffix
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import re
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
import nltk 
import spacy
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from wordcloud import WordCloud, ImageColorGenerator,STOPWORDS
import gensim
from gensim.parsing.preprocessing import STOPWORDS as gensim_stopwords
import copy
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords,wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer,LancasterStemmer
from sklearn.metrics import f1_score
from pprint import pprint
import pickle
from flask import Flask, render_template, request
import timeit
from timeit import default_timer as timer
from datetime import timedelta
import gensim.downloader
import os
app = Flask(__name__)

os.chdir(os.getcwd())
print(os.getcwd())
# isot_full_df = pd.read_csv("ISOT_Combined_FullData.csv")
# isot_full_df = isot_full_df.drop(columns = ['title','text', 'subject','date','title_length','body_length','date'])
# # Clean data using the built in cleaner in gensim
# isot_full_df['cleantext'] = isot_full_df['fulltext'].apply(lambda x: gensim.utils.simple_preprocess(x))

# X=isot_full_df['cleantext']
# y=isot_full_df['class']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y, random_state=0)

path_to_model = "Embeddings/GoogleNews-vectors-negative300.bin"
from gensim.models.keyedvectors import KeyedVectors
w2v_model = KeyedVectors.load_word2vec_format(path_to_model, binary=True)

#Inspect the model
word2vec_vocab = list(w2v_model.index_to_key)
word2vec_vocab_lower = [item.lower() for item in word2vec_vocab]



# Creating a feature vector by averaging all embeddings for all sentences
def embedding_feats(list_of_lists):
    DIMENSION = 300
    zero_vector = np.zeros(DIMENSION)
    feats = []
    for tokens in list_of_lists:
        feat_for_this =  np.zeros(DIMENSION)
        count_for_this = 0 + 1e-5 # to avoid divide-by-zero 
        for token in tokens:
            if token in w2v_model:
                feat_for_this += w2v_model[token]
                count_for_this +=1
        if(count_for_this!=0):
            feats.append(feat_for_this/count_for_this) 
        else:
            feats.append(zero_vector)
    return feats


#train_vectors = embedding_feats(X_train)
#test_vectors = embedding_feats(X_test)


#Loading the model
model_path="models/"


def fake_news_det(news,modeltyp):
    #print(type(news))
    news = pd.Series(news)
    #print(type(abc_series))
    input_data = news
    df2=pd.DataFrame()
    cleantext = news.apply(lambda x: gensim.utils.simple_preprocess(x))
    vectorized_input_data = embedding_feats(cleantext)
    if modeltyp=='RF':
        filename = model_path+"isot_ml_RF_word2vec.sav"
        clf = pickle.load(open(filename, 'rb'))
    elif modeltyp=='SVM':
        filename = model_path+"isot_ml_SVM_word2vec.sav"
        clf = pickle.load(open(filename, 'rb'))
    elif modeltyp=='KNN':
        filename = model_path+"isot_ml_KNN_word2vec.sav"
        clf = pickle.load(open(filename, 'rb'))
    elif modeltyp=='XG':
        filename = model_path+"isot_ml_XG_word2vec.sav"
        clf = pickle.load(open(filename, 'rb'))
    elif modeltyp=='NB':
        filename = model_path+"isot_ml_NB_word2vec.sav"
        clf = pickle.load(open(filename, 'rb'))
    elif modeltyp=='LR':
        filename = model_path+"isot_ml_LR_word2vec.sav"
        clf = pickle.load(open(filename, 'rb'))
    else:
        clf=''
    prediction = clf.predict(vectorized_input_data)
    return prediction

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        modeltyp = request.form['mdlselect']
        pred = fake_news_det(message,modeltyp)
        print('prediction----',pred)
        return render_template('index.html', prediction=pred)
    else:
        return render_template('index.html', prediction="Something went wrong")

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5100,debug=True)