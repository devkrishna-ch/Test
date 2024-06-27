from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from lazypredict.Supervised import LazyClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import Init
import numpy as np

# Load the features and labels for the deepfake-MIT dataset
deepfake_mit_features = Init.extract_facenet_features('Testing/Deepfakes')
deepfake_mit_labels = ['Deepfake'] * len(deepfake_mit_features)

real_features = Init.extract_facenet_features('Testing/Real')
real_labels = ['Real'] * len(real_features)


facenet_weights = np.load('facenet_weights.npy')
facenet_labels = np.load('facenet_labels.npy')

X_train, X_test, y_train, y_test = train_test_split(
    facenet_weights, facenet_labels, test_size=0.2, random_state=42
)

# Concatenate the deepfake-MIT features and labels with the testing set
X_test = np.concatenate((X_test, deepfake_mit_features, real_features))
y_test = np.concatenate((y_test, deepfake_mit_labels, real_labels))

clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)

# Fit and predict with Facenet features
models, predictions = clf.fit(X_train, X_test, y_train, y_test)

print("LazyClassifier Results:")
print(predictions)
print(type(predictions))
print(models)

# Basic plot
model_names = models.index
accuracies = models['Balanced Accuracy']

# Plot bar chart
plt.figure(figsize=(10, 6))
plt.bar(model_names, accuracies, color='skyblue')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Accuracy of Different Models')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Confusion matrix related stuff
# best_model_name = models.index[0]

# # Get the predictions of the best model from the first row of the models DataFrame
# best_model_predictions = predictions.iloc[0]

# # Print classification report
# print("\nClassification Report:")
# print(len(y_test))
# # print(len(best_model_predictions))

# print(classification_report(y_test, best_model_predictions))

# # Plot confusion matrix
# plt.figure(figsize=(8, 6))
# cm = confusion_matrix(y_test, best_model_predictions) 
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Deepfake', 'Real'], yticklabels=['Deepfake', 'Real'])
# plt.xlabel('Predicted Labels')
# plt.ylabel('True Labels')
# plt.title('Confusion Matrix')
# plt.show()
