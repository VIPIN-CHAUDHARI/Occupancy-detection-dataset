import numpy as np
import pandas as pd
#loading the dataset
df=pd.read_csv('Occupancy.csv',dtype=float)
print(df)
#1
from sklearn.preprocessing import StandardScaler
convert = StandardScaler()
feature = df.drop(['Occupied'], axis = 1)
label = df.Occupied
feature = convert.fit_transform(feature)

#2
from sklearn.model_selection import train_test_split
f_train, f_test, l_train, l_test = train_test_split(feature, label, random_state = 1, test_size = 0.3)
print('Train Data',f_train)
print('Test Data',f_test)
#3
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(criterion = 'entropy', n_estimators=3,random_state=1,n_jobs=2)
forest.fit(f_train,l_train)
y_predict= forest.predict(f_test)
train_acuracy =forest.score(f_train,l_train)
test_acc=forest.score(f_test,l_test)
print('Train accuracy',train_acuracy)
print('Test Accuracy',test_acc)
from sklearn.metrics import mean_squared_error
mns=mean_squared_error(l_test,y_predict)
print('Mean Square Error',mns)
