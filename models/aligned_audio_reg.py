import json
import os,sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pathlib
import numpy as np
from aligned_audio_reg_model import AlignClassModel
import numpy as np 
import pandas as pd
import pickle
import delta.compat as tf
import datetime
from tqdm import tqdm
from dataloader import get_reg_data
from statistics import mean 
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
# import tf.keras.optimizers.Adam as Adam
import scipy.stats as stats
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import sys


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


ckpt_save_dir = os.path.join("model_checkpoints/","aligned_audio_reg/")
pred_save_dir = os.path.join("model_predictions/","aligned_audio_reg/")
pathlib.Path(ckpt_save_dir).mkdir(parents=True, exist_ok=True) 
pathlib.Path(pred_save_dir).mkdir(parents=True, exist_ok=True) 


batch_size = 32
epochs = 1
learning_rate = 0.001
optimizer_name = 'adam'

def train(duration,labels_train,labels_val,labels_test,units,dropout,optimizer,flag):
    model_save_path = os.path.join(ckpt_save_dir,'mdl_wts_'+str(duration)+'.hdf5')
    mcp_save = tf.keras.callbacks.ModelCheckpoint(model_save_path, save_best_only=True, monitor='val_loss', mode='min')

    input_text = tf.keras.layers.Input(shape = [516,300] )
    input_speech = tf.keras.layers.Input(shape = [520,26] )
        
    output,embedding =   AlignClassModel(dropout=dropout,units = units)([input_text,input_speech])


    model = tf.keras.Model(inputs =[input_text,input_speech] ,outputs = [output])
    
    if optimizer=='adam':
        optim = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    elif optimizer =='adadelta':
        optim = tf.keras.optimizers.Adadelta(learning_rate=1.0, rho=0.95)
    model.compile( optimizer=optim,loss='mean_squared_error')
    

    if flag=='best_val':
        history = model.fit([X_train_text,X_train_audio],labels_train,batch_size=batch_size,validation_data=([X_val_text,X_val_audio],labels_val),verbose=1,epochs=epochs,callbacks=[mcp_save])
    
    if flag=='last_epoch':
        history = model.fit([X_train_text,X_train_audio],labels_train,batch_size=batch_size,validation_data=([X_val_text,X_val_audio],labels_val),verbose=1,epochs=epochs)

    if flag=='best_val':
        model.load_weights(model_save_path)

    model_path  = os.path.join(ckpt_save_dir,'mdl_'+str(duration))
    tf.saved_model.save(model, model_path)
    model.save_weights(model_save_path)
    test_loss = model.evaluate([X_test_text,X_test_audio],labels_test,batch_size=batch_size)
    train_loss = model.evaluate([X_train_text,X_train_audio],labels_train,batch_size=batch_size)

    # print("Train loss for {duration} days : {train_loss}".format(duration = duration,train_loss = train_loss))
    # print("Test loss for {duration} days : {test_loss}".format(duration = duration,test_loss = test_loss))

    pred = model.predict([X_test_text,X_test_audio])
    df  = pd.DataFrame(pred.tolist())
    pred_save_path = os.path.join(pred_save_dir,'pred_'+str(duration)+'.csv')
    df.to_csv(pred_save_path)
    model2 =   tf.keras.Model(inputs = model.input,outputs= model.get_layer(model.layers[-1].name).output[1])
    model2_path = os.path.join(ckpt_save_dir,'mdl2_'+str(duration))
    tf.saved_model.save(model2, model2_path)

    pearson_r ,_= stats.pearsonr(np.squeeze(pred), labels_test)
    tau,_ = stats.kendalltau(np.squeeze(pred),labels_test)
    rho,_ = stats.spearmanr(np.squeeze(pred),labels_test)

    return [train_loss,test_loss,tau,rho,pearson_r]



val_3  = train(3,y_train3days,y_val3days,y_test3days,units=100,dropout=0.6,optimizer='adam',flag='last_epoch')
val_7  = train(7,y_train7days,y_val7days,y_test7days,units=100,dropout=0.4,optimizer='adam',flag='last_epoch')
val_15 = train(15,y_train15days,y_val15days,y_test15days,units=100,dropout=0.45,optimizer='adam',flag='last_epoch')
val_30 = train(30,y_train30days,y_val30days,y_test30days,units=100,dropout=0.4,optimizer='adam',flag='last_epoch')


# print("MSE_TRAIN MSE_TEST  TAU  PEARSON_R   RHO")

# print("3 days :",val_3)
# print("7 days :",val_7)
# print("15 days :",val_15)
# print("30 days :",val_30)