from sklearn.metrics import confusion_matrix
from sklearn import svm

def image_classification(train_images_features, train_images_labels, test_image_features):
    # linear = svm.SVC(kernel='linear', C=1, decision_function_shape='ovo')
    # linear = linear.fit(train_images_features, train_images_labels)
    # return linear.predict(test_image_features)

    # rbf = svm.SVC(kernel='rbf', gamma=1, C=1, decision_function_shape='ovo')
    # rbf = rbf.fit(train_images_features, train_images_labels)
    # return rbf.predict(test_image_features)

    # poly = svm.SVC(kernel='poly', degree=3, C=1, decision_function_shape='ovo')
    # poly = poly.fit(train_images_features, train_images_labels)
    # return poly.predict(test_image_features)

    sig = svm.SVC(kernel='sigmoid', C=1, decision_function_shape='ovo').fit(train_images_features, train_images_labels)
    sig = sig.fit(train_images_features, train_images_labels)
    return sig.predict(test_image_features)
    
    
    