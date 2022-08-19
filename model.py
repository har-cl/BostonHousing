import pandas as pd
import numpy as np
import scipy.stats as stats
import sklearn
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle

boston = load_boston()
bos = pd.DataFrame(boston.data)

bos.columns = boston.feature_names
bos['MEDV'] = boston.target

bos['CRIM'] = np.log(bos['CRIM'])
bos['DIS'] = np.log(bos['DIS'])

x = bos.drop(columns=['MEDV'])
y = bos['MEDV']

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)

lr4 = LinearRegression()
lr4.fit(x_train, y_train)

print(lr4.score(x_train, y_train))
print(lr4.score(x_test, y_test))
print(lr4.predict([[ 2.27003508e-01,  1.61978401e-02, -1.51748086e-03,  3.00210078e+00,
       -2.20672216e+01,  4.59829803e+00, -2.32160370e-02, -7.39694444e+00,
        1.61104857e-01, -1.17746434e-02, -8.46729568e-01,  1.30103158e-02,
       -5.41207887e-01]]))

lr4_p = pickle.dump(lr4, open('linr_model.sav', 'wb'))
