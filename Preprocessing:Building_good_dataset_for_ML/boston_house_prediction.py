import pandas as pd
from sklearn.model_selection import train_test_split

df=pd.read_csv('boston_house_prices.csv')
print(df.head())

print(df.info())

print(df.isna().sum())

X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

print(X_test.shape)

### Feature scaling
### 1. Min max approach

# x-x_min/x_max-x_min

from sklearn.preprocessing import MinMaxScaler
mm=MinMaxScaler()
X_train__norm=mm.fit_transform(X_train)
X_test_norm=mm.fit_transform(X_test)

### 2. Standard Scalar
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train_norm=scaler.fit_transform(X_train)
X_test_norm=scaler.fit_transform(X_test)
