from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV, SGDClassifier,LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd
# list of tuples: the first element is a string, the second is an object
estimators = [('LogisticRegression', LogisticRegression()),('RidgeClassifier', RidgeClassifier()), ('RidgeClassifierCV', RidgeClassifierCV()),\
              ('RandomForestClassifier', RandomForestClassifier()), ('GradientBoostingClassifier', GradientBoostingClassifier())]
from sklearn.model_selection import train_test_split

data = pd.read_csv('HR_Data.csv')

# Convert all nominal to numeric.
data['sales'].replace(['sales', 'accounting', 'hr', 'technical', 'support', 'management',
        'IT', 'product_mng', 'marketing', 'RandD'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], inplace = True)
data['salary'].replace(['low', 'medium', 'high'], [0, 1, 2], inplace = True)
###########################################
# Train & Test Data

data_X = data.copy()
data_y = data_X['left']
del data_X['left']

train_X, test_X, train_y, test_y = train_test_split(data_X, data_y, test_size = 0.2, random_state = 1234)

print("Train Dataset rows: {} ".format(train_X.shape[0]))
print("Test Dataset rows: {} ".format(test_X.shape[0]))
for estimator in estimators:
    scores = cross_val_score(estimator=estimator[1],
                            X=train_X,
                            y=train_y,
                            cv=3,
                            n_jobs=-1)
    print scores