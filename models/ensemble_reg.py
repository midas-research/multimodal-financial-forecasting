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

text_df3= pd.read_csv("model_predictions/bilstm_reg/pred_3.csv")
text_df7= pd.read_csv("model_predictions/bilstm_reg/pred_7.csv")
text_df15= pd.read_csv("model_predictions/bilstm_reg/pred_15.csv")
text_df30= pd.read_csv("model_predictions/bilstm_reg/pred_30.csv")


audio_df3=pd.read_csv("model_predictions/aligned_audio_reg/pred_3.csv")
audio_df7=pd.read_csv("model_predictions/aligned_audio_reg/pred_7.csv")
audio_df15=pd.read_csv("model_predictions/aligned_audio_reg/pred_15.csv")
audio_df30=pd.read_csv("model_predictions/aligned_audio_reg/pred_30.csv")

finance_df3=pd.read_csv("model_predictions/SVR/pred_3.csv")
finance_df7=pd.read_csv("model_predictions/SVR/pred_7.csv")
finance_df15=pd.read_csv("model_predictions/SVR/pred_15.csv")
finance_df30=pd.read_csv("model_predictions/SVR/pred_30.csv")


y_test3=[]
y_test7=[]
y_test15=[]
y_test30=[]

testdf=pd.read_csv("../data/test_data.csv")

for index,row in testdf.iterrows():
    y_test3.append(float(row['future_3']))
    y_test7.append(float(row['future_7']))
    y_test15.append(float(row['future_15']))
    y_test30.append(float(row['future_30']))
    


    

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

pred_save_dir = os.path.join("model_predictions/","ensemble_reg/")
pathlib.Path(pred_save_dir).mkdir(parents=True, exist_ok=True) 


ratio_range=np.linspace(0,1,51)
mse_list = []
alpha_list=[]
beta_list=[]

def combined(duration,y_pred1,y_pred2,y_pred3,y_test):
    

    min_mse=10
    min_alpha=2
    min_beta=2
    
    for alpha in ratio_range:
        for beta in ratio_range:
            if beta<=(1-alpha):
                y_pred_combined=(alpha)*y_pred1+(beta)*y_pred2+(1-alpha-beta)*y_pred3
                mse = mean_squared_error(y_test, y_pred_combined)
                mse_list.append(mse)
                alpha_list.append(alpha)
                beta_list.append(beta)

                if mse<min_mse:
                    min_mse=mse
                    min_alpha=alpha
                    min_beta=beta                  

    # print("Min MSE for"+str(duration)+"days=" +str(min_mse))
    # print("Ratio at Min MSE: alpha="+str(min_alpha)+" beta="+str(min_beta))
    y_pred_max=(min_alpha)*y_pred1+(min_beta)*y_pred2+(1-min_alpha-min_beta)*y_pred3
    df  = pd.DataFrame(y_pred_max)
    pred_save_path = os.path.join(pred_save_dir,'pred_'+str(duration)+'.csv')
    df.to_csv(pred_save_path)
    return

combined(duration=3,y_pred1=audio_pred3,y_pred2=text_pred3,y_pred3=finance_pred3,y_test=y_test3)
combined(duration=7,y_pred1=audio_pred7,y_pred2=text_pred7,y_pred3=finance_pred7,y_test=y_test7)
combined(duration=15,y_pred1=audio_pred15,y_pred2=text_pred15,y_pred3=finance_pred15,y_test=y_test15)
combined(duration=30,y_pred1=audio_pred30,y_pred2=text_pred30,y_pred3=finance_pred30,y_test=y_test30)


