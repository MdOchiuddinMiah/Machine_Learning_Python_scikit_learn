from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_breast_cancer 
from sklearn.model_selection import train_test_split

cancer=load_breast_cancer()

x_train,x_test,y_train,y_test=train_test_split(cancer.data, cancer.target, random_state=0)

print(y_test)

mlp=MLPClassifier(random_state=42)
mlp.fit(x_train,y_train)

print('Accuracy of the training set: {:.2f}'.format(mlp.score(x_train,y_train)*100)+ ' %')
print('Accuracy of the test set: {:.2f}'.format(mlp.score(x_test,y_test)*100)+' %')



