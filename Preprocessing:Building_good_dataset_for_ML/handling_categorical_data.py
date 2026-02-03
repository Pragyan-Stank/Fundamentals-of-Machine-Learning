import pandas as pd 

df=pd.DataFrame([
    ['green','M',10.1,'class1'],
    ['red','L',13.5,'class2'],
    ['blue','XL',15.3,'class1']
])

df.columns=['color','size','price','classlabel']

print(df.head())

### Nominal Vs Ordinal features

# Nominal features -> color, price
# Ordinal features -> size

size_mapping={
    'XL':3,
    'L':2,
    'M':1
}

df['size']=df['size'].map(size_mapping)

from sklearn.preprocessing import OneHotEncoder

ohe=(OneHotEncoder(sparse_output=False).set_output(transform='pandas'))
F_=ohe.fit_transform(df[['color']])

df=pd.concat(objs=[df,F_],axis=1).drop(columns=['color'])

from sklearn.preprocessing import LabelEncoder
class_label=LabelEncoder()
y=class_label.fit_transform(df['classlabel'].values)
df['classlabel']=y

print(df)
