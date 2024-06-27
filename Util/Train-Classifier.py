from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import joblib
import numpy as np

facenet_weights = np.load('facenet_weights.npy')
facenet_labels = np.load('facenet_labels.npy')

clf = QuadraticDiscriminantAnalysis()
clf.fit(facenet_weights, facenet_labels)

joblib.dump(clf, 'qda_classifier_weights.joblib')