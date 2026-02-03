# Data Processing and Imputation Techniques

import pandas as pd
import numpy as np

df= pd.DataFrame([
    [1.0,2.0,3.0,4.0],
    [12.2,2.3,4.3,2.3],
    [np.nan,3.4,2.4,np.nan],
    [np.nan,np.nan,np.nan,np.nan]
])

df.columns=['A','B','C','D']

print(df)
print(df.isnull().sum())

df_1 = df.dropna(axis=0) ## drop row
print(df_1)

df_2 = df.dropna(axis=1) ## drop column
print(df_2)

## Only drop rows when all columns are nan

df_3 = df.dropna(axis=0, how='all')
print(df_3)


# Only drop rows where nan appears in specific column
df_4 = df.dropna(subset=['D'], axis=0)
print(df_4)

from sklearn.impute import SimpleImputer 
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')  #mean is taken column wise 

imputed_data=imputer.fit_transform(df)
print(imputed_data)

imputer1 = SimpleImputer(missing_values=np.nan, strategy='median')
imputed_data1=imputer1.fit_transform(df)
print(imputed_data1)
