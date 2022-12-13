import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import metrics
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
# %matplotlib inline -- need this?
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

#--------------------------------------------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------------------------------------#

# Grpahing
df = pd.read_csv("winequalityReds.csv")

sns.pairplot(
    df,
    x_vars=["fixed.acidity", "volatile.acidity", "citric.acid", "residual.sugar", "residual.sugar", "free.sulfur.dioxide", "free.sulfur.dioxide", "density", "density", "density", "density", "density"],
    y_vars=["quality"],
    )

#plt.show()

sns.distplot(df['quality'])

#plt.show()

sns.scatterplot(x='density', y='alcohol', data=df)

#plt.show()

x = df.drop('quality',axis=1)
y = df['quality']

#plt.show()

#--------------------------------------------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------------------------------------#

# Linear Regression

lm = LinearRegression()


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)

lm.fit(x_train,y_train)

print(lm.intercept_)

pred_train = lm.predict(x_train)

pred_test = lm.predict(x_test) # predictions

print('MAE:', metrics.mean_absolute_error(y_train, pred_train))
print('MSE:', metrics.mean_squared_error(y_train, pred_train))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_train, pred_train)))

print('MAE:', metrics.mean_absolute_error(y_test, pred_test))
print('MSE:', metrics.mean_squared_error(y_test, pred_test))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pred_test)))

fig = plt.figure()
sns.scatterplot(x=y_test, y=pred_test)
plt.show()


# print(pred_test)
# print(y_test)
#print(x_test['alcohol'])


# print("\nAccuracy")
# print("Accuracy:", )