import json
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import string
import numpy as np
import pandas as pd
import pathlib
from scipy import stats
import pickle
import sys
import string


with open('../data/audio_features1.pkl', 'rb') as f:
    audio_featDict=pickle.load(f)
    
with open('../data/audio_features2.pkl', 'rb') as f:
    audio_featDictMark2=pickle.load(f)
    
train_df = pd.read_csv("../data/train_data.csv")
test_df = pd.read_csv("../data/test_data.csv")
val_df = pd.read_csv("../data/val_data.csv")

    
with open('../data/mittens_train.pkl', 'rb') as f:
    text_train=pickle.load(f)
    
with open('../data/mittens_test.pkl', 'rb') as f:
    text_test=pickle.load(f)
    
with open('../data/mittens_val.pkl', 'rb') as f:
    text_val=pickle.load(f)

def ModifyData_clf(df,text_dict):
    error=[]
    error_text=[]
    X=[]
    X_text=[]
    y_3days=[]
    y_7days=[]
    y_15days=[]
    y_30days=[]

    for index,row in df.iterrows():
        
        try:
            
            X_text.append(text_dict[row['text_file_name'][:-9]])
         
        except:
       
            error_text.append(row['text_file_name'][:-9])

        lstm_matrix_temp = np.zeros((520, 26), dtype=np.float64)
        i=0
        
        try:
            speaker_list=list(audio_featDict[row['text_file_name'][:-9]])
            speaker_list=sorted(speaker_list, key=lambda x: (int(x.split('_')[1]), int(x.split('_')[2])))
            for sent in speaker_list:
                lstm_matrix_temp[i, :]=audio_featDict[row['text_file_name'][:-9]][sent]+audio_featDictMark2[row['text_file_name'][:-9]][sent]
                i+=1
            X.append(lstm_matrix_temp)

        except:

            Padded=np.zeros((520, 26), dtype=np.float64)
            X.append(Padded)
            error.append(row['text_file_name'][:-9])
            

        y_3days.append(float(row['lab_3']))
        y_7days.append(float(row['lab_7']))
        y_15days.append(float(row['lab_15']))
        y_30days.append(float(row['lab_30']))
        
    X=np.array(X)
    X_text=np.array(X_text)
    y_3days=np.array(y_3days)
    y_7days=np.array(y_7days)
    y_15days=np.array(y_15days)
    y_30days=np.array(y_30days)
    
    X=np.nan_to_num(X)
        
    return [X,X_text,y_3days,y_7days,y_15days,y_30days]




def ModifyData_reg(df,text_dict):
    error=[]
    error_text=[]
    X=[]
    X_text=[]
    y_3days=[]
    y_7days=[]
    y_15days=[]
    y_30days=[]

    for index,row in df.iterrows():
        
        try:
            
            X_text.append(text_dict[row['text_file_name'][:-9]])
         
        except:
       
            error_text.append(row['text_file_name'][:-9])

        lstm_matrix_temp = np.zeros((520, 26), dtype=np.float64)
        i=0
        
        try:
            speaker_list=list(audio_featDict[row['text_file_name'][:-9]])
            speaker_list=sorted(speaker_list, key=lambda x: (int(x.split('_')[1]), int(x.split('_')[2])))
            for sent in speaker_list:
                lstm_matrix_temp[i, :]=audio_featDict[row['text_file_name'][:-9]][sent]+audio_featDictMark2[row['text_file_name'][:-9]][sent]
                i+=1
            X.append(lstm_matrix_temp)

        except:

            Padded=np.zeros((520, 26), dtype=np.float64)
            X.append(Padded)
            error.append(row['text_file_name'][:-9])
            

        y_3days.append(float(row['future_3']))
        y_7days.append(float(row['future_7']))
        y_15days.append(float(row['future_15']))
        y_30days.append(float(row['future_30']))
        
    X=np.array(X)
    X_text=np.array(X_text)
    y_3days=np.array(y_3days)
    y_7days=np.array(y_7days)
    y_15days=np.array(y_15days)
    y_30days=np.array(y_30days)
    
    X=np.nan_to_num(X)
        
    return [X,X_text,y_3days,y_7days,y_15days,y_30days]


def get_reg_data():
    data_train = ModifyData_reg(train_df,text_train)

    data_test = ModifyData_reg(test_df,text_test)

    data_val = ModifyData_reg(val_df,text_val)

    return data_train, data_test, data_val

def get_clf_data():
    data_train = ModifyData_clf(train_df,text_train)

    data_test = ModifyData_clf(test_df,text_test)

    data_val = ModifyData_clf(val_df,text_val)

    return data_train, data_test, data_val