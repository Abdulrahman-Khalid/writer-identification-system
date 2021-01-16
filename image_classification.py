from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

classifiers = {
    'svm-linear': svm.SVC(kernel='linear', C=1, decision_function_shape='ovo'),
    'svm-rbf': svm.SVC(kernel='rbf', gamma=1, C=1, decision_function_shape='ovo'),
    'svm-poly': svm.SVC(kernel='poly', degree=3, C=1, decision_function_shape='ovo'),
    'svm-sigmoid': svm.SVC(kernel='sigmoid', C=1, decision_function_shape='ovo'),
    'knn': KNeighborsClassifier(n_neighbors=3),
}


def image_classification(kernel, train_images_features, train_images_labels, test_image_features):
    clsfr = classifiers[kernel]
    clsfr.fit(train_images_features, train_images_labels)
    return clsfr.predict(test_image_features)
