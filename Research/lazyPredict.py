from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier
import app1

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(app1.facenet_all_features, app1.facenet_labels, test_size=0.2, random_state=42)

# Initialize LazyClassifier
clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)

# Fit and predict with Facenet features
models, predictions = clf.fit(X_train, X_test, y_train, y_test)

print("Facenet LazyClassifier Results:")
print(predictions)