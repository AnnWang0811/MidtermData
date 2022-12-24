import joblib
model_pretrained = joblib.load('Midterm-LR-YYYYMMDD.pkl')
import pandas as pd
import numpy as np

#df_test = pd.read_csv("midterm_test.csv")
df_test = pd.read_csv("https://raw.githubusercontent.com/AnnWang0811/MidtermData/main/midterm_test.csv")

df_test.drop(['HomePlanet', 'Cabin', 'Destination', 'Name', 'Age'], axis = 1, inplace = True)

df_test['CryoSleep'].fillna(df_test['CryoSleep'].value_counts().idxmax(), inplace = True)
df_test = pd.get_dummies(data = df_test, columns=['CryoSleep'])
df_test.drop('CryoSleep_False', axis = 1, inplace = True)

df_test['VIP'].fillna(df_test['VIP'].value_counts().idxmax(), inplace = True)
df_test = pd.get_dummies(data = df_test, columns=['VIP'])
df_test.drop('VIP_False', axis = 1, inplace = True)

#df_test['Age'] = df_test.groupby(['VIP_True'])['Age'].apply(lambda x: x.fillna(x.median()))

df_test['RoomService'] = df_test.groupby(['VIP_True'])['RoomService'].apply(lambda x: x.fillna(x.median()))

df_test['FoodCourt'] = df_test.groupby(['VIP_True'])['FoodCourt'].apply(lambda x: x.fillna(x.median()))

df_test['ShoppingMall'] = df_test.groupby(['VIP_True'])['ShoppingMall'].apply(lambda x: x.fillna(x.median()))

df_test['Spa'] = df_test.groupby(['VIP_True'])['Spa'].apply(lambda x: x.fillna(x.median()))

df_test['VRDeck'] = df_test.groupby(['VIP_True'])['VRDeck'].apply(lambda x: x.fillna(x.median()))


predictions2=model_pretrained.predict(df_test)
predictions2

forSubmissionDF=pd.DataFrame(columns=['PassengerId','Transported'])

forSubmissionDF['PassengerId']=df_test['PassengerId']
prediction3 = np.array(predictions2).astype(bool)
forSubmissionDF['Transported']=prediction3

forSubmissionDF
prediction3


forSubmissionDF.to_csv('for_submission_20221224.csv', index=False)