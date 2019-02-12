import Geohash

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn import preprocessing
FILE = 'train.csv'
df = pd.read_csv(FILE)
test_df = pd.read_csv('test.csv')
category = df['Category'].unique()
district = df['PdDistrict'].unique()
crime_num = {}
crime_class = {}
class_crime = {}
district_class = {}

#统计每类罪案的个数
for crime in category:
    crime_num[crime] = len(df[df['Category'] == crime])
#地区编码
for index in range(len(district)):
    district_class[district[index]] = index
#星期几编码
day_class = {'Sunday':0,'Monday':1,'Tuesday':2,'Wednesday':3,'Thursday':4,'Friday':5,'Saturday':6}
#根据发生的次数排序罪案
crime_num = sorted(crime_num.items(),key = lambda x:x[1],reverse = True)
#从罪案到数字和从数字到罪案的对应
for index in range(len(crime_num)):
    crime_class[crime_num[index][0]] = index
    class_crime[index] = crime_num[index][0]
    
#Geohash编码
for i in df[df['Y']>=90].index.tolist():
    df['Y'][i] = 89.9999

hashXY = pd.Series([Geohash.encode(df['Y'][i],df['X'][i],precision=5) for i in df.index.tolist()])
df['hash'] = hashXY
for i in test_df[test_df['Y']>=90].index.tolist():
    test_df['Y'][i] = 89.9999
hashXY_test = [Geohash.encode(test_df['Y'][i],test_df['X'][i],precision=5) for i in test_df.index.tolist()]
test_df['hash'] = hashXY_test
for i in test_df[test_df['hash']=='9q8zpt']['Id']:
    test_df['hash'][i] = '9q8yyx'


def prepareData(frame):
	'''
	完成所有特征的编码
	'''
    def parseYear(date):
        return int(date[0:4])
    def parseMonth(date):
        return int(date[5:7])
    def parseDay(data):
        return int(data[8:10])
    def parseHour(data):
        return int(data[11:13])
    def parseClass_1(data):
        if data <= 3:
            return 0
        else:
            return 1
    frame['year'] = frame['Dates'].apply(parseYear)
    frame['month'] = frame['Dates'].apply(parseMonth)
    frame['day'] = frame['Dates'].apply(parseDay)
    frame['hour'] = frame['Dates'].apply(parseHour)
    frame['DayOfWeek'] = frame['DayOfWeek'].map(day_class)
    #frame['PdDistrict'] = frame['PdDistrict'].map(district_class)
    if 'Category' in frame.columns.values:
        frame['Category'] = frame['Category'].map(crime_class)
        frame['Class_1'] = frame['Category'].apply(parseClass_1)
    
    
    return frame

df = prepareData(df)
test_df = prepareData(test_df)

fit_index = ['PdDistrict','X','Y','year','hour','month','day','DayOfWeek','hash']
df = shuffle(df,random_state = 100)
df_train = df[fit_index]
#独热编码
df_train = pd.get_dummies(df_train)
df_test = test_df[fit_index]
df_test = pd.get_dummies(df_test)
df_train,x_test,y_train,y_test =train_test_split(df_train,df['Category'],test_size = 0.15, random_state = 100)


from sklearn.metrics import make_scorer, accuracy_score, f1_score,log_loss
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from xgboost.sklearn import XGBClassifier


clf = XGBClassifier(
learning_rate =0.1,
n_estimators=300,
max_depth=11,
min_child_weight=5,
gamma=0,
subsample=0.85,
colsample_bytree=0.8,
scale_pos_weight=1)
clf.fit(df_train,y_train)

y_pred = clf.predict_proba(x_test)
print(log_loss(y_test,y_pred))
