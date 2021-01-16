from sklearn import svm

_classifiers = {
    'linear': svm.SVC(kernel='linear', C=1, decision_function_shape='ovo'),
    'rbf': svm.SVC(kernel='rbf', gamma=1, C=1, decision_function_shape='ovo'),
    'poly': svm.SVC(kernel='poly', degree=3, C=1, decision_function_shape='ovo'),
    'sigmoid': svm.SVC(kernel='sigmoid', C=1, decision_function_shape='ovo'),
}


def image_classification(train_images_features, train_images_labels, test_image_features, kernel="sigmoid"):
    clsfr = _classifiers[kernel]
    clsfr.fit(train_images_features, train_images_labels)
    return clsfr.predict(test_image_features)
