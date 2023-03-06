import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
os.environ["RUNFILES_DIR"] = "/Users/kaansenparlak/Library/Python/3.8/share/plaidml"
os.environ["PLAIDML_NATIVE_PATH"] = "/Users/kaansenparlak/Library/Python/3.8/lib/libplaidml.dylib"

import keras
import numpy as np
from keras.models import load_model
from configs import *
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def get_sensitivity(pred, true):
    tp = np.sum(pred * true)
    fn = np.sum(abs(1-pred) * true)
    return tp/float(tp+fn)


def get_specificity(pred, true):
    tn = np.sum(abs(1-pred) * abs(1-true))
    fp = np.sum(pred * abs(1-true))
    return tn/float(tn+fp)


def get_precision(pred, true):
    tp = np.sum(pred * true)
    fp = np.sum(pred * abs(1-true))
    return tp/float(tp+fp)


def get_recall(pred, true):
    tp = np.sum(pred * true)
    fn = np.sum(abs(1 - pred) * true)
    return tp/float(tp+fn)


def get_f1score(precision, recall):
    return 2 * (precision*recall) / (precision+recall)


### Load data
x_train = np.load("np/x_train.npy")
y_train = np.load("np/y_train.npy")
x_val = np.load("np/x_val.npy")
y_val = np.load("np/y_val.npy")


x_train = keras.backend.constant(x_train)
y_train = keras.backend.constant(y_train)

x_train = x_train.eval()
y_train = y_train.eval()

x_val = keras.backend.constant(x_val)
y_val = keras.backend.constant(y_val)

x_val = x_val.eval()
y_val = y_val.eval()

auc_scores = []

val_to_fit = x_val
train_to_fit = x_train

### Load the model you want here after the training !!!
model = load_model('models/t2/' + "weights.89-0.65.hdf5")

predictions_val = model.predict(val_to_fit)

np.set_printoptions(precision=2, suppress=True)
predictions_val = np.array(predictions_val)
print(np.c_[predictions_val, y_val])
print(np.mean(abs(predictions_val-y_val)))
np.save("np/pred_val", predictions_val)

predictions_train = model.predict(train_to_fit)

np.set_printoptions(precision=2, suppress=True)
predictions_train = np.array(predictions_train)
print(np.c_[predictions_train, y_train])
print(np.mean(abs(predictions_train-y_train)))
np.save("np/pred_train", predictions_train)

plt.figure(1)
plt.plot(y_val, marker='o', markerfacecolor='blue', markersize=9)
plt.plot(predictions_val, marker='o', markerfacecolor='red', markersize=9)
plt.show()

plt.figure(2)
plt.plot(y_train, marker='o', markerfacecolor='blue', markersize=9)
plt.plot(predictions_train, marker='o', markerfacecolor='red', markersize=9)
plt.show()

predictions_val = np.squeeze(predictions_val)
predictions_train = np.squeeze(predictions_train)
print(np.sum(y_val == (np.round(predictions_val))))
print(np.sum(y_train == (np.round(predictions_train))))

fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_val, predictions_val, drop_intermediate=False)
auc_keras = auc(fpr_keras, tpr_keras)

plt.figure(3)
plt.plot(fpr_keras, tpr_keras)
plt.gca().set_aspect("equal")
plt.show()
print(auc_keras)

fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_train, predictions_train, drop_intermediate=False)
auc_keras = auc(fpr_keras, tpr_keras)

plt.figure(4)
plt.plot(fpr_keras, tpr_keras)
plt.gca().set_aspect("equal")
plt.show()
print(auc_keras)

print(get_sensitivity(np.round(predictions_val), y_val))
print(get_specificity(np.round(predictions_val), y_val))
pr_val = get_precision(np.round(predictions_val), y_val)
re_val = get_recall(np.round(predictions_val), y_val)
print(pr_val)
print(re_val)
print(get_f1score(pr_val, re_val))

print(get_sensitivity(np.round(predictions_train), y_train))
print(get_specificity(np.round(predictions_train), y_train))
pr_train = get_precision(np.round(predictions_train), y_train)
re_train = get_recall(np.round(predictions_train), y_train)
print(pr_train)
print(re_train)
print(get_f1score(pr_train, re_train))


