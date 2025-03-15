import numpy as np
from collections import Counter
import pandas as pd

def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum((x1 - x2) ** 2))
    return distance

# lets create a class for KNN
class KNN:
    def __init__(self, k=3):
        self.k = k
        
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        
    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions
    
    def _predict(self, x):
        # Compute the distances
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        
        # Get k nearest samples, labels
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # I imported the Counter data structure to make it easier to count the most common class label
        most_common = Counter(k_nearest_labels).most_common(1)[0][0]
        return most_common


df_train = pd.read_csv('MNIST_training.csv')
X_train = df_train.iloc[:, 1:].values
y_train = df_train.iloc[:, 0].values
    
df_test = pd.read_csv('MNIST_test.csv')
X_test = df_test.iloc[:, 1:].values
y_test = df_test.iloc[:, 0].values
    
knn = KNN(k=3)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
# Lets make our predictions into a numpy array to better help with our classifications
predictions_array = np.array(predictions)

correct_classif = np.sum(predictions_array == y_test)
incorrect_classif = len(y_test) - correct_classif

# Now lets generate our accuracy
accuracy = np.sum(predictions == y_test) / len(y_test)


print("K count:", knn.k)    
print("Predictions:", predictions)
print("Correct Classifications:", correct_classif)
print("Incorrect Classifications:", incorrect_classif)
print("Accuracy:", accuracy)