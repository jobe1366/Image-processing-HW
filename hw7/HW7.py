import numpy as np
from matplotlib import pyplot as plt
from HodaDataset.HodaDatasetReader import read_hoda_cdb, read_hoda_dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
import cv2 as cv
from skimage.feature import local_binary_pattern, hog
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

# Load Hoda Farsi Digit Dataset
print('loading data..')
train_images, train_labels = read_hoda_cdb('./HodaDataset/DigitDB/Train 60000.cdb')
test_images, test_labels = read_hoda_cdb('./HodaDataset/DigitDB/Test 20000.cdb')

#
def draw_confusion_matrix(y_true, y_pred, classes=None, normalize=True, title=None, cmap=plt.cm.Blues):
    acc = np.sum(y_true == y_pred) / len(y_true)
    print('Accuracy = {}'.format(acc))

    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print('Confusion Matrix = \n{}'.format(np.round(cm, 3)))

    if classes is None:
        classes = [str(i) for i in range(len(np.unique(y_true)))]

    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def fit_and_resize_image(src_images, dst_image_size):
    X = []
    for i, image in enumerate(src_images):
        X.append(np.array(cv.resize(image, (dst_image_size,dst_image_size), interpolation = cv.INTER_AREA)))
    return X

#
def extract_geometrical_features(images):
    features = np.zeros((len(images), 3))
    for i in range(len(images)):
        img = images[i]
        ret,thresh = cv.threshold(img,127,255,0)
        contours, hierarchy = cv.findContours(thresh, 1, 2)
        cnt = contours[0]
        for i in range(len(contours) - 1):
            cnt = np.concatenate((cnt, contours[i+1]), axis = 0)
        area = cv.contourArea(cnt)
        perimeter = cv.arcLength(cnt,True)
        hull = cv.convexHull(cnt)
        (x,y),(MA,ma),angle = cv.fitEllipse(cnt)
        area_hull = cv.contourArea(hull)
        features[i, 0] = (4 * np.pi * area) / np.square(perimeter) ## compactness
        features[i, 1] = area / area_hull ## solidity
        features[i, 2] = ecce = np.sqrt(1 - np.square(MA / ma)) ## eccentricity
    preProcess = preprocessing.StandardScaler().fit(features)
    features = preProcess.transform(features)
    return features


#
def extract_textural_features(images):
    # HOG and LBP
    list_hog = []
    list_lbp = []
    for feature in images:
      fd = hog(feature, orientations=9, pixels_per_cell=(8,8),cells_per_block=(3,3))
      list_hog.append(fd)

      out = local_binary_pattern(feature, P=8, R=1)
      hist, aa = np.histogram(out, bins=256)
      list_lbp.append(hist)

    hog_features = np.array(list_hog, 'float64')
    lbp_features = np.array(list_lbp, 'float64')

    preProcess_hog = preprocessing.MaxAbsScaler().fit(hog_features)
    features_hog = preProcess_hog.transform(hog_features)

    preProcess_lbp = preprocessing.MaxAbsScaler().fit(lbp_features)
    features_lbp = preProcess_lbp.transform(lbp_features)

    return features_hog, features_lbp


# resize data
print('resizing data..')
train_img = fit_and_resize_image(train_images, 32)
test_img = fit_and_resize_image(test_images, 32)

# extract features
print('features..')
all_data = train_img + test_img
## geometric features
geo = extract_geometrical_features(np.array(all_data))
geo_train_features = geo[:60000]
geo_test_features = geo[60000:]
## texture features
hg, lbp = extract_textural_features(np.array(all_data))
hg_train_features = hg[:60000]
hg_test_features = hg[60000:]
lbp_train_features = lbp[:60000]
lbp_test_features = lbp[60000:]
tx_train_features = np.concatenate((hg_train_features, lbp_train_features), axis=1)
tx_test_features = np.concatenate((hg_test_features, lbp_test_features), axis=1)


## models with Geometric features
print('models with Geometric features:')
print('Linear SVM + Geometric')
model1 = LinearSVC().fit(geo_train_features, train_labels)
test_predictions_1 = model1.predict(geo_test_features)
draw_confusion_matrix(test_labels, test_predictions_1, title='Linear SVM + Geometric')
print('KNN + Geometric')
model2 = KNeighborsClassifier(n_neighbors=7).fit(geo_train_features, train_labels)
test_predictions_2 = model2.predict(geo_test_features)
draw_confusion_matrix(test_labels, test_predictions_2, title='KNN + Geometric')
print('RBF SVM + Geometric')
model3 = SVC(kernel ='rbf', gamma = 'scale', C=10).fit(geo_train_features, train_labels)
test_predictions_3 = model3.predict(geo_test_features)
draw_confusion_matrix(test_labels, test_predictions_3, title='RBF SVM + Geometric')
print('logistic reg + Geometric')
model4 = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial').fit(geo_train_features, train_labels)
test_predictions_4 = model4.predict(geo_test_features)
draw_confusion_matrix(test_labels, test_predictions_4, title='logistic reg + Geometric')
print('MLP + Geometric')
model5 = MLPClassifier(activation='relu', hidden_layer_sizes=(200, 100), alpha = 0.3).fit(geo_train_features, train_labels)
test_predictions_5 = model5.predict(geo_test_features)
draw_confusion_matrix(test_labels, test_predictions_5, title='MLP + Geometric')
plt.show()

## models with HOG features
print('models with HOG features:')
print('Linear SVM + HOG')
model1 = LinearSVC().fit(hg_train_features, train_labels)
test_predictions_1 = model1.predict(hg_test_features)
draw_confusion_matrix(test_labels, test_predictions_1, title='Linear SVM + HOG')
print('KNN + HOG')
model2 = KNeighborsClassifier(n_neighbors=7).fit(hg_train_features, train_labels)
test_predictions_2 = model2.predict(hg_test_features)
draw_confusion_matrix(test_labels, test_predictions_2, title='KNN + HOG')
print('RBF SVM + HOG')
model3 = SVC(kernel = 'rbf', gamma = 'scale', C=10).fit(hg_train_features, train_labels)
test_predictions_3 = model3.predict(hg_test_features)
draw_confusion_matrix(test_labels, test_predictions_3, title='RBF SVM + HOG')
print('logistic reg + HOG')
model4 = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial').fit(hg_train_features, train_labels)
test_predictions_4 = model4.predict(hg_test_features)
draw_confusion_matrix(test_labels, test_predictions_4, title='logistic reg + HOG')
print('MLP + HOG')
model5 = MLPClassifier(activation='relu', hidden_layer_sizes=(200, 100), alpha = 0.3).fit(hg_train_features, train_labels)
test_predictions_5 = model5.predict(hg_test_features)
draw_confusion_matrix(test_labels, test_predictions_5, title='MLP + HOG')
plt.show()

## models with LBP features
print('models with LBP features:')
print('Linear SVM + LBP')
model1 = LinearSVC().fit(lbp_train_features, train_labels)
test_predictions_1 = model1.predict(lbp_test_features)
draw_confusion_matrix(test_labels, test_predictions_1, title='Linear SVM + LBP')
print('KNN + LBP')
model2 = KNeighborsClassifier(n_neighbors=7).fit(lbp_train_features, train_labels)
test_predictions_2 = model2.predict(lbp_test_features)
draw_confusion_matrix(test_labels, test_predictions_2, title='KNN + LBP')
print('RBF SVM + LBP')
model3 = SVC(kernel = 'rbf', gamma = 'scale', C=10).fit(lbp_train_features, train_labels)
test_predictions_3 = model3.predict(lbp_test_features)
draw_confusion_matrix(test_labels, test_predictions_3, title='RBF SVM + LBP')
print('logistic reg + LBP')
model4 = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial').fit(lbp_train_features, train_labels)
test_predictions_4 = model4.predict(lbp_test_features)
draw_confusion_matrix(test_labels, test_predictions_4, title='logistic reg + LBP')
print('MLP + LBP')
model5 = MLPClassifier(activation='relu', hidden_layer_sizes=(200, 100), alpha = 0.3).fit(lbp_train_features, train_labels)
test_predictions_5 = model5.predict(lbp_test_features)
draw_confusion_matrix(test_labels, test_predictions_5, title='MLP + LBP')
plt.show()

## models with textural features
print('models with textural features:')
print('Linear SVM + textural')
model1 = LinearSVC().fit(tx_train_features, train_labels)
test_predictions_1 = model1.predict(tx_test_features)
draw_confusion_matrix(test_labels, test_predictions_1, title='Linear SVM + textural')
print('KNN + textural')
model2 = KNeighborsClassifier(n_neighbors=7).fit(tx_train_features, train_labels)
test_predictions_2 = model2.predict(tx_test_features)
draw_confusion_matrix(test_labels, test_predictions_2, title='KNN + textural')
print('RBF SVM + textural')
model3 = SVC(kernel = 'rbf', gamma = 'scale', C=10).fit(tx_train_features, train_labels)
test_predictions_3 = model3.predict(tx_test_features)
draw_confusion_matrix(test_labels, test_predictions_3, title='RBF SVM + textural')
print('logistic reg + textural')
model4 = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial').fit(tx_train_features, train_labels)
test_predictions_4 = model4.predict(tx_test_features)
draw_confusion_matrix(test_labels, test_predictions_4, title='logistic reg + textural')
print('MLP + textural')
model5 = MLPClassifier(activation='relu', hidden_layer_sizes=(200, 100), alpha = 0.3).fit(tx_train_features, train_labels)
test_predictions_5 = model5.predict(tx_test_features)
draw_confusion_matrix(test_labels, test_predictions_5, title='MLP + textural')
plt.show()
