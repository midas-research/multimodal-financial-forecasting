import json
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import string
import numpy as np
import pandas as pd
from datetime import datetime
from dataloader import get_clf_data
import multiprocessing
import keras
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import f1_score
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation , Masking,Bidirectional,Input,concatenate,Reshape,AveragePooling1D
from keras.layers.embeddings import Embedding
from keras.layers import BatchNormalization
import pathlib
from scipy import stats
import pickle
import sys
import re
from keras.models import load_model
import regex as re
from keras.models import Model
import tensorflow as tf
from keras.utils import np_utils
from nltk.corpus import stopwords
import pickle
from sklearn.metrics import matthews_corrcoef

duration_list=[]
batch_sizes=[]
epochs_list=[]
optimizer_list=[]
training_loss_list=[]
test_loss_list=[]
pearson_list=[]
spearman_list=[]

data_train, data_test, data_val = get_clf_data()
X_train_audio,X_train_text,y_train3days, y_train7days, y_train15days, y_train30days = data_train
X_test_audio,X_test_text, y_test3days, y_test7days, y_test15days, y_test30days = data_test
X_val_audio,X_val_text,y_val3days, y_val7days, y_val15days, y_val30days = data_val

def get_audio_aligned_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

def get_audio_aligned_output(model,text_input,speech_input):
    return model.predict([text_input,speech_input])


def get_embeddings(audio_aligned_reg,text_input,speech_input):
    audio_reg_embeddings = get_audio_aligned_output(audio_aligned_reg,text_input,speech_input)
    return audio_reg_embeddings


def get_data(audio_aligned_reg):
    train_data = get_embeddings(audio_aligned_reg,X_train_text,X_train_audio)
    test_data = get_embeddings(audio_aligned_reg,X_test_text,X_test_audio)
    val_data = get_embeddings(audio_aligned_reg,X_val_text,X_val_audio)

    return train_data,val_data,test_data

def get_class(pred):
    labels = (pred > 0.5).astype(np.int)
    return labels


audio_aligned_reg_3 = get_audio_aligned_model("model_checkpoints/aligned_audio_reg/mdl2_3")
audio_aligned_reg_7 = get_audio_aligned_model("model_checkpoints/aligned_audio_reg/mdl2_7")
audio_aligned_reg_15 = get_audio_aligned_model("model_checkpoints/aligned_audio_reg/mdl2_15")
audio_aligned_reg_30 = get_audio_aligned_model("model_checkpoints/aligned_audio_reg/mdl2_30")

ckpt_save_dir = os.path.join("model_checkpoints/","aligned_audio_clf/")
pred_save_dir = os.path.join("model_predictions/","aligned_audio_clf/")
pathlib.Path(ckpt_save_dir).mkdir(parents=True, exist_ok=True) 
pathlib.Path(pred_save_dir).mkdir(parents=True, exist_ok=True) 

batch_size = 32
epochs = 1
learning_rate = 0.001
optimizer_name = 'adam'


def train_speech(duration, labels_train, labels_val, labels_test, optimizer, flag, aligned_audio_model):

    train_inp,val_inp,test_inp =get_data(aligned_audio_model)
    train_speech_input = [train_inp,np.zeros_like(train_inp)]
    val_speech_input = [val_inp,np.zeros_like(val_inp)]
    test_speech_input = [test_inp,np.zeros_like(test_inp)]

    model_save_path = os.path.join(ckpt_save_dir,'mdl_wts_'+str(duration)+'.hdf5')
    mcp_save = keras.callbacks.callbacks.ModelCheckpoint(model_save_path, save_best_only=True, monitor='val_loss', mode='min')

    units = 200
    input_speech = Input(shape=[520,26])
    input_speech_hidden_state = Input(shape=[units])
    input_speech_cell_state = Input(shape=[units])

    masked_input = Masking(mask_value=0)(input_speech)
    lstm_output = LSTM(units,dropout=0.4,recurrent_dropout=0.4)(masked_input,initial_state = [input_speech_hidden_state,input_speech_cell_state])
    dropped_lstm = Dropout(0.4)(lstm_output)
    output = Dense(1,activation='sigmoid')(dropped_lstm)

    model = Model(inputs = [input_speech,input_speech_hidden_state,input_speech_cell_state],outputs = output )

    
    if optimizer=='adam':
        optim = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
    elif optimizer =='adadelta':
        optim = keras.optimizers.Adadelta(learning_rate=1, rho=0.95)
    model.compile(loss='binary_crossentropy', optimizer=optim )
    
    if flag=='best_val':
        history = model.fit([X_train_audio,train_speech_input[0],train_speech_input[1]],labels_train,batch_size=batch_size,validation_data=([X_val_audio,val_speech_input[0],val_speech_input[0]],labels_val),verbose=1,epochs=epochs,callbacks=[mcp_save])
    
    if flag=='last_epoch':
        history = model.fit([X_train_audio,train_speech_input[0],train_speech_input[1]],labels_train,batch_size=batch_size,validation_data=([X_val_audio,val_speech_input[0],val_speech_input[0]],labels_val),verbose=1,epochs=epochs)


    if flag=='best_val':
        model.load_weights(model_save_path)

    model_path = os.path.join(ckpt_save_dir,'mdl_wts_'+str(duration)+'.h5')
    model.save(model_path)
    model.save_weights(model_save_path)
    
    test_pred = model.predict([X_test_audio,test_speech_input[0],test_speech_input[1]])
    train_pred = model.predict([X_train_audio,train_speech_input[0],train_speech_input[1]])

    test_pred = get_class(test_pred)
    train_pred = get_class(train_pred)

    test_f1 = f1_score(labels_test,test_pred)
    train_f1 = f1_score(labels_train,train_pred)

    # print("Test F1 score for {duration} days : {test_f1}".format(duration = duration,test_f1 = test_f1))
    # print("Train F1 score for {duration} days : {train_f1}".format(duration = duration,train_f1 = train_f1))

    pred = model.predict([X_test_audio,test_speech_input[0],test_speech_input[1]])
    df  = pd.DataFrame(pred.tolist())
    pred_save_path = os.path.join(pred_save_dir,'pred_'+str(duration)+'.csv')
    df.to_csv(pred_save_path)


    return [train_f1,test_f1]

val_3_speech  = train_speech(3,y_train3days,y_val3days,y_test3days,optimizer='adam',flag='best_val', aligned_audio_model=audio_aligned_reg_3)
# print("Train_F1        Test_F1")
# print("3 days :",val_3_speech)

val_7_speech  = train_speech(7,y_train7days,y_val7days,y_test7days,optimizer='adam',flag='best_val', aligned_audio_model=audio_aligned_reg_7)
# print("Train_F1        Test_F1")
# print("7 days :",val_7_speech)

val_15_speech  = train_speech(15,y_train15days,y_val15days,y_test15days,optimizer='adam',flag='best_val', aligned_audio_model=audio_aligned_reg_15)
# print("Train_F1        Test_F1")
# print("15 days :",val_15_speech)


val_30_speech  = train_speech(30,y_train30days,y_val30days,y_test30days,optimizer='adam',flag='best_val', aligned_audio_model=audio_aligned_reg_30)
# print("Train_F1        Test_F1")
# print("30 days :",val_30_speech)




