from sklearn.model_selection import train_test_split
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
FILE = 'train.csv'
df = pd.read_csv(FILE)
df = shuffle(df)

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
    frame['PdDistrict'] = frame['PdDistrict'].map(district_class)
    if 'Category' in frame.columns.values:
        frame['Category'] = frame['Category'].map(crime_class)
        #第一个预分类器的分类标签
        frame['Class_1'] = frame['Category'].apply(parseClass_1)
    
    
    return frame

df = prepareData(df)

#处理异常值
df = df[df['X']<-122.357037]
df = df[df['X']>-122.528152]
df = df[df['Y']>37.703090]
df = df[df['Y']<37.815376]

test_df = prepareData(test_df)

#测试集划分
df, x_test, y_train, y_test = train_test_split(df,df['Category'],test_size = 0.15)

#预分类之后的两类数据
df_major = df[df['Class_1'] == 0]
df_minor = df[df['Class_1'] == 1]

def prepareMinorData(frame):
	'''
	生成第二个预分类器的分类标签
	'''
    def parseClass_2(data):
        if data <=12:
            return 0
        else:
            return 1
    frame['Class_2'] = frame['Category'].apply(parseClass_2)
    return frame
    
df_minor  = prepareMinorData(df_minor)
#第二个预分类器分类过后的两类数据
df_minor_1 = df_minor[df_minor['Class_2']==0]
df_minor_2 = df_minor[df_minor['Class_2']==1]

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score
from sklearn.utils import shuffle

#所有分类器选取的特征
index_1 = ['DayOfWeek','PdDistrict','X','Y','year','hour']
index_2 = ['PdDistrict','X','Y','hour']
index_3 = ['DayOfWeek','PdDistrict','X','Y','year','hour']
index_4 = ['DayOfWeek','PdDistrict','X','Y','year','hour','month']
index_5 = ['DayOfWeek','PdDistrict','X','Y','year','hour','month','day']

print(df[['DayOfWeek','PdDistrict','X','Y','year','month','day','hour','Category']].head(5))

#预分类器1               
clf_1 = RandomForestClassifier(max_depth = 50,n_estimators = 120,max_features = 'sqrt')
#预分类器2
clf_2 = RandomForestClassifier(max_depth = 100,n_estimators = 120,max_features = 'sqrt')
#4分类器
clf_3 = RandomForestClassifier(max_depth = 160,min_samples_split = 50,n_estimators = 120,max_features = 'sqrt')
#9分类器
clf_4 = RandomForestClassifier(max_depth = 20,n_estimators = 120,max_features = 'sqrt')
#26分类器
clf_5 = RandomForestClassifier(max_depth = 20,min_samples_split = 20,n_estimators = 130,max_features = 'sqrt')

#训练模型
clf_1.fit(df[index_1],df['Class_1'])
clf_2.fit(df_minor[index_2],df_minor['Class_2'])
clf_3.fit(df_major[index_3],df_major['Category'])
clf_4.fit(df_minor_1[index_4],df_minor_1['Category'])
clf_5.fit(df_minor_2[index_5],df_minor_2['Category'])

#计算测试集上的概率
result_1 = clf_1.predict_proba(x_test[index_1])
result_2 = clf_2.predict_proba(x_test[index_2])
result_3 = clf_3.predict_proba(x_test[index_3])
result_4 = clf_4.predict_proba(x_test[index_4])
result_5 = clf_5.predict_proba(x_test[index_5])

#生成测试集上的预测结果
y_pred = []
for index in range(len(x_test)):
    if result_1[index][0]> result_1[index][1]:
        y_pred.append(result_3[index].argmax())
    else:
        if result_2[index][0]>result_2[index][0]:
            y_pred.append(result_4[index].argmax()+4)
        else:
            y_pred.append(result_5[index].argmax()+13)
print(f1_score(y_pred,y_test),average = 'micro')


    
