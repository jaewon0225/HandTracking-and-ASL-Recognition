from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import DataFormatter as dp

# Assuming you have your input vectors stored in 'X' and corresponding labels in 'y'
X,y = dp.prepare_dataset(300,30,"processed_data")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a KNN classifier with 5 neighbors
knn = KNeighborsClassifier(n_neighbors=5)

# Train the KNN classifier
knn.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# It's important to use binary mode
knnPickle = open('knnpickle_file', 'wb')
# source, destination
pickle.dump(knn, knnPickle)

# close the file
knnPickle.close()

