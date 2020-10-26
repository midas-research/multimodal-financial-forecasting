import nltk
from nltk import sent_tokenize,word_tokenize
import numpy as np
from nltk.corpus import stopwords
import string
import pandas as pd
import re
import json 
import os 
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import spacy 
import mittens
from mittens import Mittens
import nltk
from nltk import sent_tokenize,word_tokenize
import regex as re
from contractions import CONTRACTION_MAP
from nltk.corpus import stopwords
nlp = spacy.load('en_core_web_sm')
from collections import Counter
import sys
import pickle
import csv
from nltk import tokenize
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt 
import multiprocessing
import pathlib
from scipy import stats


nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
nltk.download('punkt')


def gen_sentence(row):
    path = os.path.join("../../data/earning_calls_data/",row.text_file_name)
    with open(path) as pearl:
        text = pearl.read()

    sentences = text.splitlines()
    return sentences



data_train = pd.read_csv("../../data/train_data.csv") 
data_test = pd.read_csv("../../data/test_data.csv") 
data_val = pd.read_csv("../../data/val_data.csv") 

data_train["text"] = data_train.apply(lambda row : gen_sentence(row) ,axis =1)
data_test["text"] = data_test.apply(lambda row : gen_sentence(row) ,axis =1)
data_val["text"] = data_val.apply(lambda row : gen_sentence(row) ,axis =1)



data_train.to_json('../../data/train_pk.json',orient='index')
data_test.to_json('../../data/test_pk.json',orient='index')
data_val.to_json('../../data/val_pk.json',orient='index')



data_train_clf = pd.read_csv("../../data/train_data.csv") 
data_test_clf = pd.read_csv("../../data/test_data.csv") 
data_val_clf = pd.read_csv("../../data/val_data.csv") 

data_train_clf["text"] = data_train_clf.apply(lambda row : gen_sentence(row) ,axis =1)
data_test_clf["text"] = data_test_clf.apply(lambda row : gen_sentence(row) ,axis =1)
data_val_clf["text"] = data_val_clf.apply(lambda row : gen_sentence(row) ,axis =1)



data_train_clf.to_json('../../data/train_pk_clf.json',orient='index')
data_test_clf.to_json('../../data/test_pk_clf.json',orient='index')
data_val_clf.to_json('../../data/val_pk_clf.json',orient='index')



def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())                       
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
        
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text

def intersection(lst1, lst2): 
    return list(set(lst1) & set(lst2))

def union(lst1, lst2): 
    return list(set(lst1) | set(lst2))

def get_pragmatic_tokens(lmd_path, so_cal_path, hedge_path):
    lmd_dict_negative = dict.fromkeys(pd.read_excel(lmd_path,sheet_name='Negative').iloc[:,0].tolist(),1)
    lmd_dict_positive = dict.fromkeys(pd.read_excel(lmd_path,sheet_name='Positive').iloc[:,0].tolist(),1)
    lmd_dict_uncertainity = dict.fromkeys(pd.read_excel(lmd_path,sheet_name='Uncertainty').iloc[:,0].tolist(),1)
    lmd_dict_litigious = dict.fromkeys(pd.read_excel(lmd_path,sheet_name='Litigious').iloc[:,0].tolist(),1)
    lmd_dict_strong_modal = dict.fromkeys(pd.read_excel(lmd_path,sheet_name='StrongModal').iloc[:,0].tolist(),1)
    lmd_dict_weak_modal = dict.fromkeys(pd.read_excel(lmd_path,sheet_name='WeakModal').iloc[:,0].tolist(),1)
    lmd_dict_constraining = dict.fromkeys(pd.read_excel(lmd_path,sheet_name='Constraining').iloc[:,0].tolist(),1)

    hedge_words_dic = dict.fromkeys(pd.read_csv(hedge_path).iloc[:,0].tolist(),1)
    hedge_words_dic = {key.upper():value for key,value in hedge_words_dic.items()}

    so_cal_dict = json.load(open(so_cal_path))
    so_cal_pos = {key.upper():value for key,value in so_cal_dict.items() if value > 0}
    so_cal_neg = {key.upper():value for key,value in so_cal_dict.items() if value < 0}

    combined_dict = {**lmd_dict_negative,**lmd_dict_positive,**lmd_dict_uncertainity,**lmd_dict_litigious,**lmd_dict_strong_modal,**lmd_dict_weak_modal,**lmd_dict_constraining,**hedge_words_dic,**so_cal_pos,**so_cal_neg}

    pragmatic_list = list(combined_dict.keys())
    pragmatic_list = [word.lower() for word in pragmatic_list]
    return pragmatic_list

def get_vocab_from_text(text_data_path, stock_list_path):

    df = pd.read_csv(stock_list_path)
    doc = []
    vocab = {}

    for index, row in df.iterrows():
        path = os.path.join(text_data_path,row.text_file_name)    
        with open(path) as pearl:
            text = pearl.read()
            text = expand_contractions(text)
            tokens = word_tokenize(text)
            tokens = [w.lower() for w in tokens]
            words = [w for w in tokens if not w in string.punctuation]

            words = [word for word in words if word not in stop_words and len(word)>2]
            
            for word in words:
                if word in vocab.keys():
                    vocab[word] = vocab[word]+1
                else: 
                    vocab[word] = 1

    df = pd.DataFrame.from_dict(vocab, orient='index',columns=['count'])
    vocab = list(vocab.keys())

    vocab_total = df.index.values
    df = df.sort_values(by='count')[-5000:]
    vocab_top_5000 = df.index.values

    return vocab_top_5000, vocab_total

def get_cooccurence_matrix(vocab_path, text_data_path, stock_list_path):
    df = pd.read_csv(vocab_path)

    vocabulary = dict(zip(df.iloc[:,1],range(0,len(df))))
    length =len(vocabulary)
    cooccurence_matrix = m = np.zeros([length,length]) 

    df = pd.read_csv(stock_list_path)


    for index, row in df.iterrows():
        path = os.path.join(text_data_path,row.text_file_name)    
        with open(path) as pearl:
            text = pearl.read()
            text = expand_contractions(text)
            tokens = word_tokenize(text)
            
            tokens = [w.lower() for w in tokens]
            words = [w for w in tokens if not w in string.punctuation]

            words = [word for word in words if word not in stop_words and len(word)>2]
            
            for i in range(len(words)):
                for j in range(max(i-10,0),min(i+10,len(words)-1)):
                    if j==i:
                        continue
                    elif(words[i] in vocabulary.keys() and words[j] in vocabulary.keys()):

                        index_i = vocabulary[words[i]]
                        index_j = vocabulary[words[j]]
                        cooccurence_matrix[index_i][index_j] = cooccurence_matrix[index_i][index_j] + abs((1/(j-i)))
    return cooccurence_matrix

def glove2dict(glove_filename ):
    with open(glove_filename) as f:
        reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)
        embed = {line[0]: np.array(list(map(float, line[1:])))
                for line in reader}
    return embed

def load_data(filename):

	data={}
	with open(filename) as f:
	  data = json.load(f)

	return data


def preprocess_text(text):
        processed_text=[]
   
        for line in text:
            text = expand_contractions(line)
            tokens = word_tokenize(text)
            tokens = [w.lower() for w in tokens]
            words = [w for w in tokens if not w in string.punctuation]

            words = [word for word in words if word not in stop_words and len(word)>2]
            processed_text.append(words)  
        return processed_text

def get_mittens(words):
    index = [vocabulary[word] for word in words]
    return [new_embeddings[i] for i in index]

def vectorize_text(processed_text):

    vectorized_text=[]
    for text in processed_text:
        words = [word for word in text if word in vocabulary.keys()]
        if len(words) >= 1:
            vectorized_text.append(np.mean(get_mittens(words), axis=0))
   

    return vectorized_text

def get_max_len(filepath):
    maxlen = 0
    data = load_data(filepath)
    input= []

    for i in range(len(data)):
        
        call_text=data[str(i)]["text"]
        processed_text=preprocess_text(call_text)
        vectorized_text=vectorize_text(processed_text)
        input.append(vectorized_text)

        for line in input:
            maxlen=max(maxlen,len(line))
    return maxlen

def create_embeddings(filepath, maxlen):

    data = load_data(filepath)
    input= []
    labels_3 = []
    labels_7 = []
    labels_15 = []
    labels_30 = []

    i = 0

    for i in range(len(data)):
        
        call_text=data[str(i)]["text"]
        labels_3.append(float(data[str(i)]["future_3"]))
        labels_7.append(float(data[str(i)]["future_7"]))
        labels_15.append(float(data[str(i)]["future_15"]))
        labels_30.append(float(data[str(i)]["future_30"]))    
        processed_text=preprocess_text(call_text)
        vectorized_text=vectorize_text(processed_text)
        input.append(vectorized_text)

        output=[]

        for line in input:
            l = np.asarray(line, dtype=np.float32)
            l=np.pad(l,((0,maxlen-len(l)),(0,0)), 'constant')
            output.append(l)
        
        output = np.array(output)
        return output



lmd_path =  '../../data/LoughranMcDonald_SentimentWordLists_2018.xlsx' 
hedge_words_path = '../../data/dictionary.csv'
so_cal_path = '../../data/so-cal.json'
stock_list_path = '../../data/train_data.csv'
text_data_path =  "../../data/earning_calls_data"
vocab_path = 'union_vocab.csv'

pragmatic_list = get_pragmatic_tokens(lmd_path, so_cal_path, hedge_words_path)
vocab_top_5000, vocab_total = get_vocab_from_text(text_data_path, stock_list_path)

pragmatic_vocab = intersection(pragmatic_list,vocab_total)
union_vocab = union(vocab_top_5000,pragmatic_vocab)
pd.DataFrame(union_vocab).to_csv('union_vocab.csv')

cooccurence_matrix = get_cooccurence_matrix(vocab_path, text_data_path, stock_list_path)
df = pd.DataFrame(cooccurence_matrix)
df.to_csv('coocur_union.csv')

vocab = pd.read_csv('union_vocab.csv')
vocabulary = dict(zip(vocab.iloc[:,1],range(0,len(vocab))))
vocab = vocabulary.keys()
cooccurrence = pd.read_csv('coocur_union.csv').iloc[:,1:].to_numpy()

mittens_model = Mittens(n=300, max_iter=1000)
original_embedding = glove2dict('../../data/glove.6B.300d.txt')
new_embeddings = mittens_model.fit(
    cooccurrence,
    vocab=vocab,
    initial_embedding_dict= original_embedding)

print("MITTENS TRAINED")
filename_train="../../data/train_pk.json"
filename_test="../../data/test_pk.json"
filename_val="../../data/val_pk.json"

maxlen = 0
maxlen = max(maxlen, get_max_len(filename_train))
maxlen = max(maxlen, get_max_len(filename_val))
maxlen = max(maxlen, get_max_len(filename_test))


pickle.dump(create_embeddings(filename_train, maxlen), open("train_union_mitten.pkl", "wb"))
pickle.dump(create_embeddings(filename_val, maxlen), open("val_union_mitten.pkl", "wb"))
pickle.dump(create_embeddings(filename_test, maxlen), open("test_union_mitten.pkl", "wb"))












