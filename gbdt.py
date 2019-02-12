import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
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
    
df = df[df['X']<-122.357037]
df = df[df['X']>-122.528152]
df = df[df['Y']>37.703090]
df = df[df['Y']<37.815376]

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


fit_index = ['PdDistrict','X','Y','year','hour','day','month','DayOfWeek']
df = shuffle(df,random_state = 100)
df_train = df[fit_index]
#独热编码
df_train = pd.get_dummies(df_train)
df_train,x_test,y_train,y_test =train_test_split(df_train,df['Category'],test_size = 0.2, random_state = 100)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score,log_loss
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
               
test_df = pd.read_csv('test.csv')
test_df = prepareData(test_df)
test_df = test_df[fit_index]
test_df = pd.get_dummies(test_df)
test_re = clf.predict_proba(test_df)
final = pd.DataFrame({class_crime[i]:[test_re[j][i] for j in range(len(test_df))] for i in range(len(category))})
final.to_csv('submission.csv',index = True)

y_pred = clf.predict_proba(x_test)
print(log_loss(y_test,y_pred))
