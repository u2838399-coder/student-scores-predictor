import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

df=pd.read_csv("studentsscore_data.csv", sep=";")

print(df.head())


df["results"] = df["G3"].apply(lambda x: 1 if x >= 10 else 0)

features=['studytime','failures','absences','G1','G2']
X=df[features]
y=df["results"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model=LogisticRegression()
model.fit(X_train,y_train)

predictions=model.predict(X_test)
print("Accuracy:",accuracy_score(y_test,predictions))

newstudent=scaler.transform([[3,0,4,12,13]])

prediction=model.predict(newstudent)

if prediction[0]==1:
    print("Student will Pass")
else:
    print("Student will Fail")

importance=model.coef_[0]
plt.bar(features,importance)
plt.title("Feature Importance")
plt.show()


