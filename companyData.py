#Get company information
#INPUT: 'symbol'
#OUTPUT: company information, printed in console,  json format
import pandas as pd 
import tensorflow as tf
import numpy as np 
import datetime
import sys
import requests
import json 

def getCompany(symbol):
    token="pk_f21b1474e546423f9a39275cb33eed97"
    URL= "https://cloud.iexapis.com/stable/stock/"+symbol+"/company/?token="+token
    URL_logo= "https://cloud.iexapis.com/stable/stock/"+symbol+"/logo/?token="+token

    r = requests.get(URL)

    r_logo = requests.get(URL_logo)


   
    data = r.json() 
    data_logo=r_logo.json()

    data.update(data_logo)
    print(data) 
  

    # file=open(r'C:\Users\aalex\Desktop\InformatiiCompanie\\'+symbol+'.txt','w+');
    # file.write(json.dumps(data1)); # json  dictionary in string 
    # file.close()

   

if __name__ == "__main__":
    arguments= sys.argv;
    getCompany(sys.argv[1]);
 