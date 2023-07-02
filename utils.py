import matplotlib.pyplot as plt
import cv2

from tensorflow.keras import backend as K
import numpy as np

def plot_from_img_path(rows, columns, list_img_path):
    fig = plt.figure(figsize=(12,12))
    for i in range(1, rows*columns +1):
        fig.add_subplot(rows, columns, i)
        img= cv2.imread(list_img_path[i])
        img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        
    plt.show()
    

def dice_coefficients(y_true, y_pred, smooth=100):
    y_true_flatten = K.flatten(y_true)
    y_pred_flatten = K.flatten(y_pred)
    
    intersection = K.sum(y_true_flatten*y_pred_flatten)
    union = K.sum(y_true_flatten) + K.sum(y_pred_flatten)
    return (2* intersection + smooth) /(union + smooth)

def dice_coefficient_loss(y_true, y_pred, smooth=100):
    return -dice_coefficients(y_true, y_pred, smooth)

def iou(y_true, y_pred, smooth=100):
    intersection = K.sum(y_true*y_pred)
    sum = K.sum(y_true + y_pred)
    iou = (intersection + smooth) / (sum -intersection + smooth)
    return iou

def jaccard_distance(y_true, y_pred):
    y_true_flatten = K.flatten(y_true)
    y_pred_flatten = K.flatten(y_pred)
    
    return -iou(y_true_flatten, y_pred_flatten)
    
    