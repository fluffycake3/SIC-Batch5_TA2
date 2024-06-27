import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#baca csv
df = pd.read_csv ("ai4i2020.csv")

#cek data
print(df.head())
#cek data kosong
print(df.isnull().sum())

#buang data yang gak penting kolom 0,1,2,-1,-2,-3,-4, -5
df2 = df.drop(df.columns[[0,1,-1,-2,-3,-4,-5]], axis=1)
#ganti Type L,M,H ke angka 0.5, 0.3, 0.2
d = {"H":0.2,'M':0.3,"L":0.5}
df2["Type"]= df2["Type"].map(d)

#cek ulang data
#print (df2)

#tetapkan target dan feature
tgt = df2["Machine failure"].values.reshape(-1,1) #target
ftr = df2.drop(["Machine failure"], axis=1).values.reshape(-1,6) #feature


# Pisahkan data test dan data model
xtrain, xtest, ytrain, ytest = train_test_split(ftr, tgt.ravel(), train_size=0.8)


#pemodelan
model = LogisticRegression()
model.fit(xtrain,ytrain)

#prediksi dan tes akurasi
pred = model.predict(xtest)
akura = accuracy_score (ytest, pred)
print("akurasi data : ", akura*100,"%")

# Contoh penggunaan dengan 5 fitur yang sesuai dengan model Logistic Regression
dt_contoh = np.array([[0.5,200, 300, 1200, 40, 50]])

# Prediksi dengan model
pred = model.predict(dt_contoh)
print("Prediksi Failure : ", pred)
