import itertools
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from utils.data_structures import stream_type


def plot_confusion_matrix(cm, class_name_list, fig_savepath, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # if normalize:
    #     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #     print("Normalized confusion matrix")
    # else:
    #     print('Confusion matrix, without normalization')

    # print(cm)
    plt.figure()
    plt.imshow(cm, interpolation='nearest', aspect='auto', cmap=cmap)
    plt.clim(0, 1)
    plt.title(title)
    #plt.colorbar()
    tick_marks = np.arange(len(class_name_list))
    plt.xticks(tick_marks, class_name_list, fontsize=7, rotation=90)
    plt.yticks(tick_marks, class_name_list, fontsize=7)

    fmt = '.1f' #if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center", verticalalignment="center",
                fontsize=5, color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(fig_savepath, dpi=300)
    plt.close()


def plot_class_detections(true_detections, false_detections, class_name_list, fig_savepath):

    width = 0.35
    fig, ax = plt.subplots()
    ax.bar(class_name_list, true_detections, width, label="True Detections")
    ax.bar(class_name_list, false_detections, width, bottom=true_detections, label="False Detections")

    ax.set_ylabel("Detecions")
    plt.xticks(rotation=90)
    plt.tight_layout()
    ax.legend(loc=9)
    plt.savefig(fig_savepath)
    plt.close()


def plot_regression_diff(str_mode, stype, swm_targets, swm_preds):

    rgb_weights_preds = flow_weights_preds = audio_weights_preds = None

    plt.figure()
    print("plotting regression diff ...")

    if stype == stream_type.rgbflow or stype == stream_type.rgbaudio or stype == stream_type.rgbflowaudio:
        rgb_weights_gt = np.array(swm_targets)[:, 0]
        rgb_weights_preds = np.array(swm_preds)[:, 0]
        print("RGB weights GT = ", rgb_weights_gt)
        print("RGB weights preds = ", rgb_weights_preds)
        if stype == stream_type.rgbflow or stype == stream_type.rgbflowaudio:
            flow_weights_gt = np.array(swm_targets)[:, 1]
            flow_weights_preds = np.array(swm_preds)[:, 1]
            print("Flow weights GT = ", flow_weights_gt)
            print("Flow weights preds = ", flow_weights_preds)
            if stype == stream_type.rgbflowaudio:
                audio_weights_gt = np.array(swm_targets)[:, 2]
                audio_weights_preds = np.array(swm_preds)[:, 2]
                print("Audio weights GT = ", flow_weights_gt)
                print("Audio weights preds = ", flow_weights_preds)
        elif stype == stream_type.rgbaudio:
            audio_weights_gt = np.array(swm_targets)[:, 1]
            audio_weights_preds = np.array(swm_preds)[:, 1]
            print("Audio weights GT = ", flow_weights_gt)
            print("Audio weights preds = ", flow_weights_preds)
    elif stype == stream_type.flowaudio:
        flow_weights_gt = np.array(swm_targets)[:, 0]
        flow_weights_preds = np.array(swm_preds)[:, 0]
        print("Flow weights GT = ", flow_weights_gt)
        print("Flow weights preds = ", flow_weights_preds)
        audio_weights_gt = np.array(swm_targets)[:, 1]
        audio_weights_preds = np.array(swm_preds)[:, 1]
        print("Audio weights GT = ", flow_weights_gt)
        print("Audio weights preds = ", flow_weights_preds)

    mse_stream = mean_squared_error(np.array(swm_targets), np.array(swm_preds)) 
    mae_stream = mean_absolute_error(np.array(swm_targets), np.array(swm_preds)) 
    std_stream = np.std(np.array(swm_targets) - np.array(swm_preds)) 
    
    return mse_stream, mae_stream, std_stream
