import os 
import csv
import pandas as pd
import sys
import numpy as np
import calendar
import json
from pathlib import Path
import warnings
warnings.filterwarnings("error")

count = 0


def get_date(year,month,day):
    if(len(month)==1):
        month = "0"+month
    
    if(len(day)==1):
        day = "0"+day
        
    date = year+month+day
    return date

def volatality_past(list_val ,anchor ,duration ):
    list_tmp = list_val[anchor+1:anchor+duration+1]
    mean_r = np.mean(list_tmp)
    vol = np.log(np.sqrt(np.sum(np.square(list_tmp - mean_r))/(duration)))
    return vol

def volatality_fut(list_val , anchor ,duration ):
    list_tmp = list_val[anchor-duration:anchor]
    mean_r = np.mean(list_tmp)
    vol = np.log(np.sqrt(np.sum(np.square(list_tmp - mean_r))/(duration)))
    return vol
    

def remove_dollar(string):
    
    try :
        return float(string.replace('$',''))
    except AttributeError :
        return float(string)
        

def calculate_vol(data_frame):
    list_val =np.asarray(data_frame.iloc[:,0].tolist())
    list_val = [remove_dollar(x)  for x in list_val]
    list_val_r = []  
    for i in range(0 , len(list_val)-1):
        try :
            list_val_r.append((list_val[i] - list_val[i+1])/list_val[i+1])
            
        except ZeroDivisionError :
            print(i,list_val[i] , list_val[i+1])
        
    
    past_3 = volatality_past(list_val_r , 30 , 3)
    past_7 = volatality_past(list_val_r, 30 , 7)
    past_15 = volatality_past(list_val_r, 30 , 15)
    past_30 = volatality_past(list_val_r, 30 , 30)
    fut_3 = volatality_fut(list_val_r , 30 ,  3)
    fut_7 = volatality_fut(list_val_r , 30 ,  7)
    fut_15 = volatality_fut(list_val_r , 30 ,  15)
    fut_30 = volatality_fut(list_val_r , 30 ,30)

    if(list_val[27]>list_val[28]):
        lab_3 = 1
    else :
        lab_3 = 0

    if(list_val[23]>list_val[24]):
        lab_7 = 1
    else :
        lab_7 = 0

    if(list_val[15]>list_val[16]):
        lab_15 = 1
    else :
        lab_15 = 0

    if(list_val[0]>list_val[1]):
        lab_30 = 1
    else :
        lab_30 = 0                        

    
    
    return past_3 , past_7 , past_15 ,past_30 , fut_3 , fut_7 , fut_15 ,fut_30, lab_3,lab_7,lab_15,lab_30
    

def get_value(row):
    file = os.path.join("../../data/AllRetPrices/",row.ticker+".csv")
    

    try:
        df = pd.read_csv(file)
    except FileNotFoundError:
        file = os.path.join("../../data/AllRetPrices/",row.ticker+".xls")
        try :
            df = pd.read_csv(file)
           
            
        except FileNotFoundError :
        
            print("File Not Found :",file)
            return [None , None , None, None, None, None]
    
    df['Date'] = pd.to_datetime(df['Date'])
    
    if(len(str(row.month))==1):
        month = "0"+str(row.month)
    else: 
        month = str(row.month)
    if(len(str(row.day))==1):
        day = "0"+str(row.day)
    else: 
        day = str(row.day)
        
    date = str(row.year)+'-'+month+'-'+day


    index = df[df['Date'] == date].index.tolist()[0]
   
      
    
    data_frame_volatality = df.iloc[index -30:index + 32 , 1:2]

    
    past_3 , past_7 , past_15 ,past_30 , future_3 , future_7 , future_15 ,future_30,lab_3,lab_7,lab_15,lab_30 = calculate_vol(data_frame_volatality)
    
    
    return [past_3 , past_7 , past_15 ,past_30 , future_3 , future_7 , future_15 ,future_30,lab_3,lab_7,lab_15,lab_30 ]
    

data = pd.read_csv("../../data/stock_data.csv")  


data["text_file_name"] = data.apply(lambda row : str(row['name'])+"_"+get_date(str(row.year),str(row.month),str(row.day))+"/Text.txt" ,axis =1)
data_tmp = data.apply(lambda row : get_value(row) ,axis =1).tolist()
data_tmp = pd.DataFrame(data_tmp , columns = ['past_3' , 'past_7' , 'past_15' ,'past_30' , 'future_3' , 'future_7' , 'future_15' ,'future_30',"lab_3","lab_7","lab_15","lab_30"])

data_final = pd.concat([data , data_tmp] , axis  = 1)
data_final = data_final.dropna()
data_final.to_csv('../../data/price_vol_data.csv',index=False)
data = pd.read_csv("../../data/price_vol_data.csv") 
data['total_days'] = data.apply(lambda row : row.month*30 + row.day , axis = 1)

data = data.sort_values(by = 'total_days')
row = len(data)
data = data.drop(columns=  ["total_days"])
train_split = int(row*0.6)
val_split =train_split+ int(row*0.2)

data_train = data[:train_split]
data_val = data[train_split:val_split]
data_test = data[val_split:]

data_train.to_csv('../../data/train_data.csv',index=False)
data_val.to_csv('../../data/val_data.csv')
data_test.to_csv('../../data/test_data.csv',index=False)



