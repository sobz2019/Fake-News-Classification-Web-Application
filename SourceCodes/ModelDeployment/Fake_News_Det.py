import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import json
import tensorflow_hub as hub
import tensorflow_text
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
from imblearn.over_sampling import SMOTE,ADASYN
from collections import Counter
import nltk 


import tensorflow as tf
import bert
from tensorflow.keras.models import  Model
from tqdm import tqdm
import tensorflow_hub as hub
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.layers import Dense, Dropout, Activation,Embedding,Flatten,LSTM,Bidirectional
from tensorflow.keras.layers import Reshape,Conv1D, MaxPooling1D, Concatenate
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.backend import clear_session

from keras.backend import clear_session
from tensorflow.keras.models import load_model
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords,wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer,LancasterStemmer
from sklearn.metrics import f1_score
import pickle
from flask import Flask, render_template, request
import warnings
warnings.filterwarnings('ignore')
import os
app = Flask(__name__)

os.chdir(os.getcwd())
print(os.getcwd())

#Loading the model
model_path="models/"

bert_layer=hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",trainable=True)

# bert preprocessor - https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3
preprocessor = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
# bert encoder - https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-512_A-8/2
encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-512_A-8/2",trainable=True)


print('Models Loaded')




def dl_create_input_array(sentences,seqlen):
    MAX_SEQ_LEN=seqlen
    input_word_ids = tf.keras.layers.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32,name="input_word_ids")
    input_mask = tf.keras.layers.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32,name="input_mask")
    segment_ids = tf.keras.layers.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32,name="segment_ids")

    
    def get_masks(tokens, max_seq_length):
        return [1]*len(tokens) + [0] * (max_seq_length - len(tokens))

    def get_segments(tokens, max_seq_length):
#         """Segments: 0 for the first sequence, 1 for the second"""
        segments = []
        current_segment_id = 0
        for token in tokens:
            segments.append(current_segment_id)
            if token == "[SEP]":
                current_segment_id = 1
        return segments + [0] * (max_seq_length - len(tokens))

    pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    #Create the tokenizer with the BERT layer and import it tokenizer using the original vocab file.
    FullTokenizer=bert.bert_tokenization.FullTokenizer
    vocab_file=bert_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case=bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer=FullTokenizer(vocab_file,do_lower_case)
    
    
    def get_ids(tokens, tokenizer, max_seq_length):
        """Token ids from Tokenizer vocab"""
        token_ids = tokenizer.convert_tokens_to_ids(tokens,)
        input_ids = token_ids + [0] * (max_seq_length-len(token_ids))
        return input_ids

    def create_single_input(sentence,MAX_LEN):
        stokens = tokenizer.tokenize(sentence)
        stokens = stokens[:MAX_LEN]
        stokens = ["[CLS]"] + stokens + ["[SEP]"]

        ids = get_ids(stokens, tokenizer, MAX_SEQ_LEN)
        masks = get_masks(stokens, MAX_SEQ_LEN)
        segments = get_segments(stokens, MAX_SEQ_LEN)

        return ids,masks,segments

    input_ids, input_masks, input_segments = [], [], []
    for sentence in tqdm(sentences,position=0, leave=True):
        ids,masks,segments=create_single_input(sentence,MAX_SEQ_LEN-2)
        input_ids.append(ids)
        input_masks.append(masks)
        input_segments.append(segments)
    return [np.asarray(input_ids, dtype=np.int32), np.asarray(input_masks, dtype=np.int32), np.asarray(input_segments, dtype=np.int32)]


def ml_embedding_feat_gen(df):
    
    # preprocessing dataset  - First Set
    inputs = preprocessor(df['fulltext'])
    # feeding it to model for vectorization
    outputs = encoder(inputs)
    pooled_output = outputs["pooled_output"]      
    sequence_output = outputs["sequence_output"]  
    # defining dataframe
    df2=pd.DataFrame()
    # Converting bert encoder sequence output to 1 dimension for ML Model training
    for i in range(0,len(sequence_output)):
        b=sequence_output[i].numpy().sum(axis=0)
        df2=df2.append(pd.Series(b),ignore_index=True)
    return df2


def clean_format(text):
    clean_text=text.replace("\n", " ")
    clean_text=re.sub('\[[^]]*\]', ' ', text)
    #clean_text = re.sub('[^a-zA-Z]',' ',clean_text)  # replaces non-alphabets with spaces
    clean_text=re.sub(r' {2,}',' ',clean_text)
    return clean_text

def fake_news_det(message,dataset,modeltyp):
    df=pd.DataFrame([message], columns=['fulltext'])
    if dataset=='ISOT':
        print('modeltyp2--->',modeltyp)
        model_path="models/"
        if modeltyp in('XgBoost','KNN','RandomForest','SVM','LogisticRegression','NaiveBayes'):
            bert_feat_df=ml_embedding_feat_gen(df)
            if modeltyp=='RandomForest':
                filename = model_path+"isot_ml_RF_bert.sav"
                print('filename--->',filename)
                clf = pickle.load(open(filename, 'rb'))
            elif modeltyp=='SVM':
                filename = model_path+"isot_ml_SVM_bert.sav"
                clf = pickle.load(open(filename, 'rb'))
            elif modeltyp=='KNN':
                filename = model_path+"isot_ml_KNN_bert.sav"
                clf = pickle.load(open(filename, 'rb'))
            elif modeltyp=='XgBoost':
                filename = model_path+"isot_ml_XG_bert.sav"
                clf = pickle.load(open(filename, 'rb'))
            elif modeltyp=='NaiveBayes':
                filename = model_path+"isot_ml_NB_bert.sav"
                clf = pickle.load(open(filename, 'rb'))
            elif modeltyp=='LogisticRegression':
                filename = model_path+"isot_ml_LR_bert.sav"
                clf = pickle.load(open(filename, 'rb'))
            else:
                clf=''
            prediction = clf.predict(bert_feat_df)
            print('prediction_ml------------>',prediction)
            return prediction
        else:
            sentences=df.fulltext.values
            sent_inputs=dl_create_input_array(sentences,256)
            if modeltyp=='CNN':
                filename = model_path+"model_ISOT_CNN_BERT.h5"
                clf = load_model(filename,custom_objects={'KerasLayer':hub.KerasLayer})
            elif modeltyp=='LSTM':
                filename = model_path+"model_ISOT_LSTM_BERT.h5"
                clf = load_model(filename,custom_objects={'KerasLayer':hub.KerasLayer})
            else:
                filename = model_path+"model_ISOT_BILSTM_BERT.h5"
                clf = load_model(filename,custom_objects={'KerasLayer':hub.KerasLayer})
            print('model_path------------>',model_path)
            pred = clf.predict(sent_inputs)
            pred = np.array((pred > 0.5).astype(int)[:,0]).tolist()
            prediction=pred[0]
            print('prediction_dl------------>',prediction)
            return prediction

    elif dataset=='FakeNewsNet':
        print('modeltyp2--->',modeltyp)
        model_path="models/fakenewsnet_bert/"
        if modeltyp in('XgBoost','KNN','RandomForest','SVM','LogisticRegression','NaiveBayes'):
            bert_feat_df=ml_embedding_feat_gen(df)
            if modeltyp=='RandomForest':
                filename = model_path+"fakenewsnet_ml_RF_bert.sav"
                print('filename--->',filename)
                clf = pickle.load(open(filename, 'rb'))
            elif modeltyp=='SVM':
                filename = model_path+"fakenewsnet_ml_SVM_bert.sav"
                clf = pickle.load(open(filename, 'rb'))
            elif modeltyp=='KNN':
                filename = model_path+"fakenewsnet_ml_KNN_bert.sav"
                clf = pickle.load(open(filename, 'rb'))
            elif modeltyp=='XgBoost':
                filename = model_path+"fakenewsnet_ml_XG_bert.sav"
                clf = pickle.load(open(filename, 'rb'))
            elif modeltyp=='NaiveBayes':
                filename = model_path+"fakenewsnet_ml_NB_bert.sav"
                clf = pickle.load(open(filename, 'rb'))
            elif modeltyp=='LogisticRegression':
                filename = model_path+"fakenewsnet_ml_LR_bert.sav"
                clf = pickle.load(open(filename, 'rb'))
            else:
                clf=''
            prediction = clf.predict(bert_feat_df)
            print('prediction_ml------------>',prediction)
            return prediction
        else:
            sentences=df.fulltext.values
            
            if modeltyp=='CNN':
                sent_inputs=dl_create_input_array(sentences,128)
                filename = model_path+"model_FAKENEWSNET_CNN_BERT_V1.h5"
                clf = load_model(filename,custom_objects={'KerasLayer':hub.KerasLayer})
            elif modeltyp=='LSTM':
                sent_inputs=dl_create_input_array(sentences,256)
                filename = model_path+"model_FAKENEWSNET_LSTM_BERT_PRETRAINED_V2.h5"
                clf = load_model(filename,custom_objects={'KerasLayer':hub.KerasLayer})
            else:
                sent_inputs=dl_create_input_array(sentences,128)
                filename = model_path+"model_FAKENEWSNET_BILSTM_BERT_PRETRAINED_V2.h5"
                clf = load_model(filename,custom_objects={'KerasLayer':hub.KerasLayer})
            pred = clf.predict(sent_inputs)
            pred = np.array((pred > 0.5).astype(int)[:,0]).tolist()
            prediction=pred[0]
            print('prediction_dl------------>',prediction)
            return prediction
    
    elif dataset=='FAKEDDIT':
        print('modeltyp2--->',modeltyp)
        model_path="models/fakeddit_bert/"
        if modeltyp in('XgBoost','KNN','RandomForest','SVM','LogisticRegression','NaiveBayes'):
            bert_feat_df=ml_embedding_feat_gen(df)
            if modeltyp=='RandomForest':
                filename = model_path+"fakeddit_RF_bert.sav"
                print('filename--->',filename)
                clf = pickle.load(open(filename, 'rb'))
            elif modeltyp=='SVM':
                filename = model_path+"fakeddit_SVM_bert.sav"
                clf = pickle.load(open(filename, 'rb'))
            elif modeltyp=='KNN':
                filename = model_path+"fakeddit_KNN_bert.sav"
                clf = pickle.load(open(filename, 'rb'))
            elif modeltyp=='XgBoost':
                filename = model_path+"fakeddit_Xgboost_bert.sav"
                clf = pickle.load(open(filename, 'rb'))
            elif modeltyp=='NaiveBayes':
                filename = model_path+"fakeddit_NB_bert.sav"
                clf = pickle.load(open(filename, 'rb'))
            elif modeltyp=='LogisticRegression':
                filename = model_path+"fakeddit_Logistic_bert.sav"
                clf = pickle.load(open(filename, 'rb'))
            else:
                clf=''
            prediction = clf.predict(bert_feat_df)
            print('prediction_ml------------>',prediction)
            return prediction
        else:
            sentences=df.fulltext.values
            if modeltyp=='CNN':
                sent_inputs=dl_create_input_array(sentences,128)
                filename = model_path+"model_FAKEDDIT_CNN_BERT_V2.h5"
                clf = load_model(filename,custom_objects={'KerasLayer':hub.KerasLayer})
            elif modeltyp=='LSTM':
                sent_inputs=dl_create_input_array(sentences,128)
                filename = model_path+"model_FAKEDDIT_BILSTM_BERT_V2.h5"
                clf = load_model(filename,custom_objects={'KerasLayer':hub.KerasLayer})
            else:
                sent_inputs=dl_create_input_array(sentences,128)
                filename = model_path+"model_FAKEDDIT_BILSTM_BERT_V2.h5"
                clf = load_model(filename,custom_objects={'KerasLayer':hub.KerasLayer})
            pred = clf.predict(sent_inputs)
            pred = np.array((pred > 0.5).astype(int)[:,0]).tolist()
            prediction=pred[0]
            print('prediction_dl------------>',prediction)
            return prediction

    else:
        pred=''
        return prediction


@app.route('/')
def home():
    print('home Loaded')
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        dataset = request.form['Dataset']
        #print(message)
        modeltyp = request.form['mdlselect']
        print('dataset----->',dataset)
        print('modeltyp----->',modeltyp)
        pred = fake_news_det(message,dataset,modeltyp)
        print('prediction----',pred)
        return render_template('index.html', prediction=pred)
    else:
        return render_template('index.html', prediction="Something went wrong")

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5100,debug=True)
    print('URL Loaded')