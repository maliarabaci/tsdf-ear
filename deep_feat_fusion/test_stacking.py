import os
import argparse
import numpy as np
import pickle
from configparser import ConfigParser

# Custom imports
from utils.data_structures import *
from utils.utility_functions import *

# Third party module imports
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


config_parser = ConfigParser()

param_parser = argparse.ArgumentParser(description='Test Stacking Algorithm for Multi-modal Egocentric Activity Recognition')
param_parser.add_argument('--config', '-c', required=True, help='configuration file path')


def test_stacking_fusion(str_config_path):

    # Parse configuration file at first
    config_parser.read(str_config_path)
    str_modality = config_parser.get('ar_common_params', 'modality')
    str_model_rootdir = config_parser.get('common_params', 'model_rootdir')

    modality = get_stream_type(str_modality)
    print("Stream selection = {}".format(str_modality))
    if modality == stream_type.rgb or modality == stream_type.flow or modality == stream_type.audio:
        raise ValueError("Need to be multistream!")

    result_dir = get_result_dir()
    result_filename = "test_result_" + str_modality + ".txt"
    result_filepath = os.path.join(result_dir, result_filename)

    with open(result_filepath, 'w+') as result_file:

        # save the best model and the corresponding scaler 
        model_filename =  str_modality + ".pkl"
        model_filepath = os.path.join(get_model_dir(str_model_rootdir, model_type.stacking), model_filename)
        # scaler_filename = "scaler_" + ktype + ".pkl"
        # scaler_filepath = os.path.join(model_dir, scaler_filename)
        # print("model filepath = ", model_filepath)
        with open(model_filepath, 'rb') as model_file:
            stacking_model = pickle.load(model_file)

        # with open(scaler_filepath, 'rb') as scaler_file:
        #     scaler = pickle.load(scaler_file)

        # Test
        test_vid_feats, test_vid_labels = get_confidence_values(modality, conf_feat_scaler=None)
        segment_label_pred = []
        segment_label_gt = []
        vid_label_pred = []
        vid_label_gt = []
        for vid_name, vid_feats in test_vid_feats.items():        
            model_vid_pred = stacking_model.predict(vid_feats).tolist()
            # Use majority voting for the predicted video label
            vid_label_pred.append(max(set(model_vid_pred), key = model_vid_pred.count))
            vid_label_gt.append(test_vid_labels[vid_name])
            # Add all segment predictions to the list to evaluate the segment performance later
            segment_label_pred.extend(model_vid_pred)
            segment_label_gt.extend((test_vid_labels[vid_name]*np.ones(len(model_vid_pred), dtype=int)).tolist())

        vid_acc, \
        vid_f1score, \
        seg_acc, \
        seg_f1score, \
        conf_mat_vid, \
        conf_mat_seg = evaluate_performance(vid_label_gt, vid_label_pred, segment_label_gt, segment_label_pred, result_dir)

        print("Video-based Accuracy for the best model = {}".format(vid_acc*100))
        print("Video-based F1 score for the best model = {}".format(vid_f1score*100))
        print("Segment-based accuracy for the best model = {}".format(seg_acc*100))
        print("Segment-based F1 score for the best model = {}".format(seg_f1score*100))
        
        class_index_file = os.path.join("data", "class_index.txt")
        class_name_list, _ = parse_classnames(class_index_file)

        result_file.write("Video-based Accuracy for the best model = {}\n".format(vid_acc*100))
        result_file.write("Video-based F1 score for the best model = {}\n".format(vid_f1score*100))
        result_file.write("Segment-based accuracy for the best model = {}\n".format(seg_acc*100))
        result_file.write("Segment-based F1 score for the best model = {}\n".format(seg_f1score*100))
        result_file.write("Video-based class accuracies = \n")
        result_file.write("{}\n".format(class_name_list))
        result_file.write("{}\n".format(conf_mat_vid.diagonal()))
        result_file.write("Video-based confusion matrix = \n")
        result_file.write("{}\n".format(class_name_list))
        result_file.write("{}\n".format(conf_mat_vid))
        result_file.write("Segment-based class accuracies = \n")
        result_file.write("{}\n".format(class_name_list))
        result_file.write("{}\n".format(conf_mat_seg.diagonal()))
        result_file.write("Segment-based confusion matrix = \n")
        result_file.write("{}\n".format(class_name_list))
        result_file.write("{}\n".format(conf_mat_seg))


def get_confidence_values(stype, conf_feat_scaler):

    dict_confidence_values = dict()
    conf_feats = dict()
    labels = None

    # Get confidence values from single stream models
    if stype == stream_type.rgbflow or stype == stream_type.rgbaudio or stype == stream_type.rgbflowaudio:
        dict_confidence_values["rgb"], rgb_vid_labels = get_single_stream_confidences(stream_type.rgb)
        vid_labels = rgb_vid_labels

    if stype == stream_type.rgbflow or stype == stream_type.flowaudio or stype == stream_type.rgbflowaudio:
        dict_confidence_values["flow"], flow_vid_labels = get_single_stream_confidences(stream_type.flow)
        if vid_labels is None:
            vid_labels = flow_vid_labels

    if stype == stream_type.rgbaudio or stype == stream_type.rgbaudio or stype == stream_type.rgbflowaudio:
        dict_confidence_values["audio"], audio_vid_labels = get_single_stream_confidences(stream_type.audio)
        if vid_labels is None:
            raise ValueError("Something is wrong during reading model confidences!")

    # Append confidence values 
    if stype == stream_type.rgbflow:
        if rgb_vid_labels != flow_vid_labels:
            raise ValueError("Label inconsistency during reading model confidences!")

        for k, v in dict_confidence_values["rgb"].items():
            conf_feats[k] = np.hstack((v, dict_confidence_values["flow"][k]))
            if conf_feat_scaler is not None:
                conf_feats[k] = conf_feat_scaler.transform(conf_feats[k])

    elif stype == stream_type.rgbaudio:
        if rgb_vid_labels != audio_vid_labels:
            raise ValueError("Label inconsistency during reading model confidences!")

        for k, v in dict_confidence_values["rgb"].items():
            conf_feats[k] = np.hstack((v, dict_confidence_values["audio"][k]))
            if conf_feat_scaler is not None:
                conf_feats[k] = conf_feat_scaler.transform(conf_feats[k])

    elif stype == stream_type.flowaudio:
        if flow_vid_labels != audio_vid_labels:
            raise ValueError("Label inconsistency during reading model confidences!")

        for k, v in dict_confidence_values["flow"].items():
            conf_feats[k] = np.hstack((v, dict_confidence_values["audio"][k]))
            if conf_feat_scaler is not None:
                conf_feats[k] = conf_feat_scaler.transform(conf_feats[k])

    elif stype == stream_type.rgbflowaudio:
        if rgb_vid_labels != flow_vid_labels or flow_vid_labels != audio_vid_labels:
            raise ValueError("Label inconsistency during reading model confidences!")

        for k, v in dict_confidence_values["rgb"].items():
            conf_feats[k] = np.hstack((np.hstack((v, dict_confidence_values["flow"][k])), dict_confidence_values["audio"][k]))
            if conf_feat_scaler is not None:
                conf_feats[k] = conf_feat_scaler.transform(conf_feats[k])

    return conf_feats, vid_labels


def get_single_stream_confidences(stype):

    print("reading confidences...")

    if stype == stream_type.rgbflow or stype == stream_type.rgbaudio or stype == stream_type.flowaudio or stype == stream_type.rgbflowaudio:
        raise ValueError("Should be single stream while reading model confidences!")

    # Get for each stream independently 
    feat_rootdir = config_parser.get('common_params', 'deepfeat_rootdir')
    feat_dir = get_feat_dir(feat_rootdir, feat_type.single)

    # All deep features should have scaler since it is expected to be trained before
    str_stream_type = get_stream_type_string(stype)
    scaler_filename = str_stream_type + "_feats_scaler.pkl"
    scaler_filepath = os.path.join(feat_dir, scaler_filename)
    with open(scaler_filepath, 'rb') as scaler_file:
        feat_scaler = pickle.load(scaler_file)
    
    # Get only the features at video level for test samples
    feat_filename = str_stream_type + "_feats_test.pkl"
    feat_filepath = os.path.join(feat_dir, feat_filename)
    with open(feat_filepath, 'rb') as feat_file:
        feats = pickle.load(feat_file)
        labels = pickle.load(feat_file)

    for k, v in feats.items():
        feats[k] = feat_scaler.transform(v)

    # Get confidences from the selected model
    conf_values = dict()
    model_filename =  str_stream_type + ".pkl"
    str_model_rootdir = config_parser.get('common_params', 'model_rootdir')
    model_filepath = os.path.join(get_model_dir(str_model_rootdir, model_type.arm), model_filename)
    print("AR model path = ", model_filepath)
    with open(model_filepath, 'rb') as model_file:
        ar_model = pickle.load(model_file)
        for k, v in feats.items():
            conf_values[k] = ar_model.predict_proba(v)

    return conf_values, labels


def evaluate_performance(vid_label_gt, vid_label_pred, segment_label_gt, segment_label_pred, result_dir):

    print("Evaluating performance...")
    print("Total nof videos used for evaluation = {}".format(len(vid_label_gt)))
    print("Total nof video segments used for evaluation = {}".format(len(segment_label_gt)))

    model_vid_accuracy = accuracy_score(vid_label_gt, vid_label_pred)
    model_vid_f1 = f1_score(vid_label_gt, vid_label_pred, average='weighted')

    conf_mat_video = confusion_matrix(vid_label_gt, vid_label_pred)

    # Evaluate segment-based performance
    model_segment_accuracy = accuracy_score(segment_label_gt, segment_label_pred)
    model_segment_f1 = f1_score(segment_label_gt, segment_label_pred, average='weighted')

    conf_mat_segment = confusion_matrix(segment_label_gt, segment_label_pred)

    # Normalize confusion matrix
    conf_mat_video = conf_mat_video.astype('float') / conf_mat_video.sum(axis=1)[:, np.newaxis]
    conf_mat_segment = conf_mat_segment.astype('float') / conf_mat_segment.sum(axis=1)[:, np.newaxis]
    
    return model_vid_accuracy, model_vid_f1, model_segment_accuracy, model_segment_f1, conf_mat_video, conf_mat_segment


def get_result_dir():

    str_result_rootdir = config_parser.get('common_params', 'result_rootdir')

    str_result_dir = os.path.join(str_result_rootdir, "stacking")

    if not os.path.exists(str_result_dir):
        os.makedirs(str_result_dir, exist_ok=True)
    print("Saving results to directory %s" % (str_result_dir))

    return str_result_dir


if __name__ == '__main__':

    global args

    print("argument parser")    
    args = param_parser.parse_args()
    test_stacking_fusion(args.config)