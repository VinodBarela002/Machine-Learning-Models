import pandas as pd
from sklearn.datasets import load_iris
iris = load_iris()
frame = pd.DataFrame(iris.data,columns=iris.feature_names)
frame.head()

frame['target'] = iris.target
frame.head()

frame0 = frame[:50]
frame1 = frame[50:100]
frame2 = frame[100:]

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('petal Length')
plt.ylabel('petal Width')
plt.scatter(frame0['petal length (cm)'], frame0['petal width (cm)'],color="red")
plt.scatter(frame1['petal length (cm)'], frame1['petal width (cm)'],color="green")

from sklearn.model_selection import train_test_split
X = frame.drop(['target'], axis='columns')
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=10)
from sklearn.svm import SVC
model = SVC()
model.fit(X_train, y_train)

model.score(X_test, y_test)

model.predict([[5.1,3.7,2.5,0.9]])

