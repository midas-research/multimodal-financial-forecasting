import json
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import nltk
import string
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt 
import multiprocessing
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation , Masking,Bidirectional
from keras.layers.embeddings import Embedding
from keras.layers import BatchNormalization
import pathlib
from scipy import stats
import pickle
import sys
import string
from dataloader import get_reg_data

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))




duration_list=[]
batch_sizes=[]
epochs_list=[]
optimizer_list=[]
training_loss_list=[]
test_loss_list=[]
pearson_list=[]
spearman_list=[]



data_train, data_test, data_val = get_reg_data()
X_train_audio,X_train_text,y_train3days, y_train7days, y_train15days, y_train30days = data_train
X_test_audio,X_test_text, y_test3days, y_test7days, y_test15days, y_test30days = data_test
X_val_audio,X_val_text,y_val3days, y_val7days, y_val15days, y_val30days = data_val


out_train  = X_train_text
out_val = X_val_text
out_test = X_test_text

maxlen = np.shape(out_train)[1]

ckpt_save_dir = os.path.join("model_checkpoints/","bilstm_reg/")
pred_save_dir = os.path.join("model_predictions/","bilstm_reg/")
pathlib.Path(ckpt_save_dir).mkdir(parents=True, exist_ok=True) 
pathlib.Path(pred_save_dir).mkdir(parents=True, exist_ok=True)

batch_size = 32
epochs = 2
learning_rate = 0.001
optimizer_name = 'adam'

def train(duration,labels_train,labels_val,labels_test,units,dropout,optimizer,flag):
    model_save_path = os.path.join(ckpt_save_dir,'mdl_wts_'+str(duration)+'.hdf5')
    mcp_save = keras.callbacks.callbacks.ModelCheckpoint(model_save_path, save_best_only=True, monitor='val_loss', mode='min')

    model = Sequential()
    model.add(Masking(mask_value=0,input_shape = [maxlen,300]))
    model.add(Bidirectional(LSTM(units, dropout=dropout, recurrent_dropout=dropout)))
    model.add(Dropout(dropout))
    model.add(Dense(1,activation='linear'))
    
    if optimizer=='adam':
        optim = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
    elif optimizer =='adadelta':
        optim = keras.optimizers.Adadelta(learning_rate=1, rho=0.95)
    model.compile( optimizer=optim,loss='mean_squared_error')
    
    if flag=='best_val':
        history = model.fit(out_train,labels_train,batch_size=batch_size,validation_data=(out_val,labels_val),verbose=1,epochs=epochs,callbacks=[mcp_save])
    
    if flag=='last_epoch':
        history = model.fit(out_train,labels_train,batch_size=batch_size,validation_data=(out_val,labels_val),verbose=1,epochs=epochs)

    if flag=='best_val':
        model.load_weights(model_save_path)

    model_path = os.path.join(ckpt_save_dir,'mdl_wts_'+str(duration)+'.h5')
    model.save(model_path)
    model.save_weights(model_save_path)
    test_loss = model.evaluate(out_test,labels_test,batch_size=batch_size)
    train_loss = model.evaluate(out_train,labels_train,batch_size=batch_size)

    # print("Train loss for {duration} days : {train_loss}".format(duration = duration,train_loss = train_loss))
    # print("Test loss for {duration} days : {test_loss}".format(duration = duration,test_loss = test_loss))

    pred = model.predict(out_test)
    df  = pd.DataFrame(pred.tolist())
    pred_save_path = os.path.join(pred_save_dir,'pred_'+str(duration)+'.csv')
    df.to_csv(pred_save_path)

    pearson_r ,_= stats.pearsonr(np.squeeze(pred), labels_test)
    tau,_ = stats.kendalltau(np.squeeze(pred),labels_test)
    rho,_ = stats.spearmanr(np.squeeze(pred),labels_test)

    return [train_loss,test_loss,tau,rho,pearson_r]




val_3  = train(3,y_train3days,y_val3days,y_test3days,units=100,dropout=0.4,optimizer='adam',flag='best_val')
val_7  = train(7,y_train7days,y_val7days,y_test7days,units=100,dropout=0.2,optimizer='adam',flag='best_val')
val_15 = train(15,y_train15days,y_val15days,y_test15days,units=100,dropout=0.2,optimizer='adam',flag='best_val')
val_30 = train(30,y_train30days,y_val30days,y_test30days,units=100,dropout=0.3,optimizer='adam',flag='best_val')



# print("MSE_TRAIN MSE_TEST  TAU  PEARSON_R   RHO")

# print("3 days :",val_3)
# print("7 days :",val_7)
# print("15 days :",val_15)
# print("30 days :",val_30)