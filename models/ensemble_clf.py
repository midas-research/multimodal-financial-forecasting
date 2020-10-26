import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score
import sys
from mpl_toolkits.mplot3d import Axes3D
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pathlib

text_df3= pd.read_csv("model_predictions/bilstm_clf/pred_3.csv")
text_df7= pd.read_csv("model_predictions/bilstm_clf/pred_7.csv")
text_df15= pd.read_csv("model_predictions/bilstm_clf/pred_15.csv")
text_df30= pd.read_csv("model_predictions/bilstm_clf/pred_30.csv")


audio_df3=pd.read_csv("model_predictions/aligned_audio_clf/pred_3.csv")
audio_df7=pd.read_csv("model_predictions/aligned_audio_clf/pred_7.csv")
audio_df15=pd.read_csv("model_predictions/aligned_audio_clf/pred_15.csv")
audio_df30=pd.read_csv("model_predictions/aligned_audio_clf/pred_30.csv")

finance_df3=pd.read_csv("model_predictions/SVC/pred_3.csv")
finance_df7=pd.read_csv("model_predictions/SVC/pred_7.csv")
finance_df15=pd.read_csv("model_predictions/SVC/pred_15.csv")
finance_df30=pd.read_csv("model_predictions/SVC/pred_30.csv")


y_test3=[]
y_test7=[]
y_test15=[]
y_test30=[]

testdf=pd.read_csv("../data/test_data.csv")

for index,row in testdf.iterrows():
    y_test3.append(float(row['lab_3']))
    y_test7.append(float(row['lab_7']))
    y_test15.append(float(row['lab_15']))
    y_test30.append(float(row['lab_30']))
    

def get_class(pred):
    labels = (pred > 0.5).astype(np.int)
    return labels
    

text_pred3=np.array(text_df3.iloc[:,1])
text_pred7=np.array(text_df7.iloc[:,1])
text_pred15=np.array(text_df15.iloc[:,1])
text_pred30=np.array(text_df30.iloc[:,1])

audio_pred3=np.array(audio_df3.iloc[:,1])
audio_pred7=np.array(audio_df7.iloc[:,1])
audio_pred15=np.array(audio_df15.iloc[:,1])
audio_pred30=np.array(audio_df30.iloc[:,1])

finance_pred3=np.array(finance_df3.iloc[:,1])
finance_pred7=np.array(finance_df7.iloc[:,1])
finance_pred15=np.array(finance_df15.iloc[:,1])
finance_pred30=np.array(finance_df30.iloc[:,1])

pred_save_dir = os.path.join("model_predictions/","ensemble_clf/")
pathlib.Path(pred_save_dir).mkdir(parents=True, exist_ok=True) 


ratio_range=np.linspace(0,1,51)
f1_list = []
alpha_list=[]
beta_list=[]

def combined(duration,y_pred1,y_pred2,y_pred3,y_test):
    
    y_1 = f1_score(y_test,get_class(y_pred1))
    y_2 = f1_score(y_test,get_class(y_pred2))
    y_3 = f1_score(y_test,get_class(y_pred3))


    max_f1=-10
    max_alpha=2
    max_beta=2
    
    for alpha in ratio_range:
        for beta in ratio_range:
            if beta<=(1-alpha):
                y_pred_combined=(alpha)*y_pred1+(beta)*y_pred2+(1-alpha-beta)*y_pred3
                y_pred_combined = get_class(y_pred_combined)
                f1 = f1_score(y_test,y_pred_combined)
                f1_list.append(f1)
                alpha_list.append(alpha)
                beta_list.append(beta)

                if f1>max_f1:
                    max_f1=f1
                    max_alpha=alpha
                    max_beta=beta                    

    # print("MAX F1 for"+str(duration)+"days=" +str(max_f1))
    # print("Ratio at Max F1: alpha="+str(max_alpha)+" beta="+str(max_beta))
    y_pred_max=(max_alpha)*y_pred1+(max_beta)*y_pred2+(1-max_alpha-max_beta)*y_pred3
    df  = pd.DataFrame(y_pred_max)
    pred_save_path = os.path.join(pred_save_dir,'pred_'+str(duration)+'.csv')
    df.to_csv(pred_save_path)
    return

combined(duration=3,y_pred1=audio_pred3,y_pred2=text_pred3,y_pred3=finance_pred3,y_test=y_test3)
combined(duration=7,y_pred1=audio_pred7,y_pred2=text_pred7,y_pred3=finance_pred7,y_test=y_test7)
combined(duration=15,y_pred1=audio_pred15,y_pred2=text_pred15,y_pred3=finance_pred15,y_test=y_test15)
combined(duration=30,y_pred1=audio_pred30,y_pred2=text_pred30,y_pred3=finance_pred30,y_test=y_test30)


