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
    token="https://www.youtube.com/watch?v=otCpCn0l4Wo"
    
    URL= "https://cloud.iexapis.com/stable/stock/"+symbol+"/company/?token="+token;
   
    r = requests.get(URL);
   
    data = r.json() 
    print(data)

    file=open(r'C:\Users\aalex\Desktop\InformatiiCompanie\\'+symbol+'.txt','w+');
    file.write(json.dumps(data1)); # json  dictionary in string 
    file.close()

   




if __name__ == "__main__":
    arguments= sys.argv;
    getCompany(sys.argv[1]);
 