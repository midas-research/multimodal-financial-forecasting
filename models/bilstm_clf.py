import json
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import string
import numpy as np
import pandas as pd
from datetime import datetime
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
import sys
import re
from dataloader import get_clf_data
from keras.models import load_model
from keras.models import Model
from keras.utils import np_utils
from nltk.corpus import stopwords
import pickle

duration_list=[]
batch_sizes=[]
epochs_list=[]
optimizer_list=[]
training_loss_list=[]
test_loss_list=[]
pearson_list=[]
spearman_list=[]

def get_class(pred):
    labels = (pred > 0.5).astype(np.int)
    return labels

def get_bilstm_embedding_model(model_path):
    bilstm_model = load_model(model_path)
    intermediate_layer_model = Model(inputs=bilstm_model.input,
                                    outputs=bilstm_model.get_layer(bilstm_model.layers[-3].name).output)

    return intermediate_layer_model

data_train, data_test, data_val = get_clf_data()
X_train_audio,X_train_text,y_train3days, y_train7days, y_train15days, y_train30days = data_train
X_test_audio,X_test_text, y_test3days, y_test7days, y_test15days, y_test30days = data_test
X_val_audio,X_val_text,y_val3days, y_val7days, y_val15days, y_val30days = data_val

bilstm_reg_3 = get_bilstm_embedding_model("model_checkpoints/bilstm_reg/mdl_wts_3.h5")
bilstm_reg_7 = get_bilstm_embedding_model("model_checkpoints/bilstm_reg/mdl_wts_7.h5")
bilstm_reg_15 = get_bilstm_embedding_model("model_checkpoints/bilstm_reg/mdl_wts_15.h5")
bilstm_reg_30 = get_bilstm_embedding_model("model_checkpoints/bilstm_reg/mdl_wts_30.h5")

ckpt_save_dir = os.path.join("model_checkpoints/","bilstm_clf/")
pred_save_dir = os.path.join("model_predictions/","bilstm_clf/")
pathlib.Path(ckpt_save_dir).mkdir(parents=True, exist_ok=True) 
pathlib.Path(pred_save_dir).mkdir(parents=True, exist_ok=True) 


batch_size = 32
epochs = 1 
learning_rate = 0.001
optimizer_name = 'adam'

def train(duration,labels_train,labels_val,labels_test,units,dropout,optimizer,flag):
    model_save_path = os.path.join(ckpt_save_dir,'mdl_wts_'+str(duration)+'.hdf5')
    mcp_save = keras.callbacks.callbacks.ModelCheckpoint(model_save_path, save_best_only=True, monitor='val_loss', mode='min')


    input_text = Input(shape=(516,300))

    if duration == 3:
        bilstm_out = bilstm_reg_3(input_text)
    elif duration == 7:
        bilstm_out = bilstm_reg_7(input_text)
    elif duration == 15:
        bilstm_out = bilstm_reg_15(input_text)
    elif duration == 30:
        bilstm_out = bilstm_reg_30(input_text)


    dropout_output = Dropout(dropout)(bilstm_out)
    output = Dense(1,activation='sigmoid')(dropout_output)
    model = Model(inputs = [input_text], outputs = output)
    
    if optimizer=='adam':
        optim = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
    elif optimizer =='adadelta':
        optim = keras.optimizers.Adadelta(learning_rate=1, rho=0.95)
    model.compile(loss='binary_crossentropy', optimizer=optim )
    
    if flag=='best_val':
        history = model.fit(X_train_text,labels_train,batch_size=batch_size,validation_data=(X_val_text,labels_val),verbose=1,epochs=epochs,callbacks=[mcp_save])
    
    if flag=='last_epoch':
        history = model.fit(X_train_text,labels_train,batch_size=batch_size,validation_data=(X_val_text,labels_val),verbose=1,epochs=epochs)


    if flag=='best_val':
        model.load_weights(model_save_path)

    model_path = os.path.join(ckpt_save_dir,'mdl_wts_'+str(duration)+'.h5')
    model.save(model_path)
    model.save_weights(model_save_path)
    
    test_pred = model.predict(X_test_text)
    train_pred = model.predict(X_train_text)

    test_pred = get_class(test_pred)
    train_pred = get_class(train_pred)

    test_f1 = f1_score(labels_test,test_pred)
    train_f1 = f1_score(labels_train,train_pred)

    # print("Test F1 score for {duration} days : {test_f1}".format(duration = duration,test_f1 = test_f1))
    # print("Train F1 score for {duration} days : {train_f1}".format(duration = duration,train_f1 = train_f1))

    pred = model.predict(X_test_text)
    df  = pd.DataFrame(pred.tolist())
    pred_save_path = os.path.join(pred_save_dir,'pred_'+str(duration)+'.csv')
    df.to_csv(pred_save_path)

    return [train_f1,test_f1]


val_3  = train(3,y_train3days,y_val3days,y_test3days,units=100,dropout=0.4,optimizer='adam',flag='last_epoch')
val_7  = train(7,y_train7days,y_val7days,y_test7days,units=100,dropout=0.4,optimizer='adam',flag='last_epoch')
val_15  = train(15,y_train15days,y_val15days,y_test15days,units=100,dropout=0.3,optimizer='adam',flag='last_epoch')
val_30  = train(30,y_train30days,y_val30days,y_test30days,units=100,dropout=0.4,optimizer='adam',flag='last_epoch')



# print("F1_TRAIN F1_TEST ")

# print("3 days :",val_3)
# print("7 days :",val_7)
# print("15 days :",val_15)
# print("30 days :",val_30)


