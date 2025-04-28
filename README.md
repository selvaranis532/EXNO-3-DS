## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
df = pd.read_csv("/content/Encoding Data.csv")
df
```

![Screenshot 2025-04-22 104543](https://github.com/user-attachments/assets/648b9654-48aa-44e2-adb8-8ddf97bf81dc)

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm = ['Hot','Warm','Cold'] # Removed the extra space at the beginning of this line.
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![Screenshot 2025-04-22 104840](https://github.com/user-attachments/assets/680dd988-1cd5-45e2-b484-41f0a5127c99)

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![Screenshot 2025-04-22 105147](https://github.com/user-attachments/assets/21e14a9e-89f4-4799-888a-845276ff3b59)

```
le=LabelEncoder()
dfc = df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```

![Screenshot 2025-04-22 105601](https://github.com/user-attachments/assets/2267aacc-cf66-4d81-902d-828bb3e1f4eb)

```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2=pd.concat([df2,enc],axis=1)
df2
```

![Screenshot 2025-04-22 105715](https://github.com/user-attachments/assets/07c09df1-38f0-4157-ac21-9bf9cdebe2d1)
```
pd.get_dummies(df2,columns=["nom_0"])
```
![image](https://github.com/user-attachments/assets/991afe74-4aad-4c59-a3c5-3b20a64c81b6)

```
pip install --upgrade category_encoders
```

![Screenshot 2025-04-22 110003](https://github.com/user-attachments/assets/ae7907e1-0976-4691-abb6-57a9671749e5)

```
import pandas as pd
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df
```

![Screenshot 2025-04-22 110110](https://github.com/user-attachments/assets/f751fa2f-51f8-4f60-8247-b1723b9d35f3)

```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df
```

![Screenshot 2025-04-22 110605](https://github.com/user-attachments/assets/d6643a72-1d05-40fa-8002-5adcf96b3cbd)

```
dfb=pd.concat([df,nd],axis=1)
dfb
```

![Screenshot 2025-04-22 110648](https://github.com/user-attachments/assets/8f297a08-4c4f-4998-ba5b-bb2beb5146b4)

```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```

![Screenshot 2025-04-22 110728](https://github.com/user-attachments/assets/b1f017cd-38dd-4233-8ed8-c5cae9c71587)


```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df
```

![image](https://github.com/user-attachments/assets/3f40949c-b5e9-4ba4-ac94-3e4d24f0d2d8)

```
df.skew()
```

![Screenshot 2025-04-22 111228](https://github.com/user-attachments/assets/b70ca86f-4e70-46b3-b215-14c7259a8632)

```
np.log(df["Highly Positive Skew"])
```

![image](https://github.com/user-attachments/assets/6ebb863a-b897-49ea-a1e1-bdf784598adc)

```
np.reciprocal(df["Moderate Positive Skew"])
```

![image](https://github.com/user-attachments/assets/48b2e0ec-d78b-45d6-930e-2d2625c4df2e)

```
np.sqrt(df["Highly Positive Skew"])
```

![image](https://github.com/user-attachments/assets/2deb168a-4a55-44a0-a7b3-6d60ad3ec7d7)
```
np.square(df["Highly Positive Skew"])
```

![image](https://github.com/user-attachments/assets/6cdea150-4eb3-4eb1-a52c-ca513ac17718)

```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```

![image](https://github.com/user-attachments/assets/9de2f483-1f70-4bee-b3c5-2711e8d88c50)

```
df.skew()
```

![image](https://github.com/user-attachments/assets/caebbaf4-78fc-4c49-b929-8c4bc9174bd5)

```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```

![image](https://github.com/user-attachments/assets/de1c67d6-0968-4928-af9e-c7afb401b6ec)

```
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

df = pd.read_csv("Data_to_Transform.csv")  
qt = QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"] = qt.fit_transform(df[["Moderate Negative Skew"]])
df
```

![image](https://github.com/user-attachments/assets/04f2dfd7-b033-4c84-888e-1f1838f26ca5)


```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/22c5ba1f-84e9-40e7-a098-1078e26cb7cb)


```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/cedfe634-bb25-45af-92f6-414494db07d7)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/b222546b-cbf0-485e-81ce-e7bddb58f4ba)

```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/05d90e36-d1ea-424d-b861-456ae4ade0d9)

```
dt=pd.read_csv("/content/titanic_dataset (3).csv")
dt
```

![image](https://github.com/user-attachments/assets/12bf9443-ff62-46f2-9338-d0166b9731ed)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45') 
plt.show()
```

![image](https://github.com/user-attachments/assets/60c3d19c-1a74-4f0c-9ddd-dcf380f53a87)

```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/fd4e7f47-5600-4df0-9849-0d7c7169c781)

# RESULT:
       Thus the given data, Feature Encoding, Transformation process and save the data to a file
    was performed successfully
       
