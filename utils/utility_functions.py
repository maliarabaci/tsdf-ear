import os
import pickle
import numpy as np
from collections import defaultdict
from utils.data_structures import stream_type, model_type


def get_model_dir(str_model_rootdir, model_type_val):
    
    if model_type_val == model_type.arm:
        str_model_dir = os.path.join(str_model_rootdir, "arm")
    elif model_type_val == model_type.stacking:
        str_model_dir = os.path.join(str_model_rootdir, "stacking")
    elif model_type_val == model_type.swm:
        str_model_dir = os.path.join(str_model_rootdir, "tsdf", "swm")
    elif model_type_val == model_type.dwm:
        str_model_dir = os.path.join(str_model_rootdir, "tsdf", "dwm")
    else:
        raise ValueError("Undefined model type!")

    return str_model_dir


def load_ar_models(str_model_rootdir, stream_type_val):

    dict_ar_models = defaultdict(dict)

    if stream_type_val == stream_type.rgbflow or stream_type_val == stream_type.rgbaudio or stream_type_val == stream_type.rgbflowaudio:
        rgb_model_filepath = get_model_filepath(str_model_rootdir, model_type.arm, stream_type.rgb)
        with open(rgb_model_filepath, 'rb') as rgb_model_file:
            dict_ar_models['rgb'] = pickle.load(rgb_model_file)

    if stream_type_val == stream_type.rgbflow or stream_type_val == stream_type.flowaudio or stream_type_val == stream_type.rgbflowaudio:
        flow_model_filepath = get_model_filepath(str_model_rootdir, model_type.arm, stream_type.flow)
        with open(flow_model_filepath, 'rb') as flow_model_file:
            dict_ar_models['flow'] = pickle.load(flow_model_file)

    if stream_type_val == stream_type.rgbaudio or stream_type_val == stream_type.flowaudio or stream_type_val == stream_type.rgbflowaudio:
        audio_model_filepath = get_model_filepath(str_model_rootdir, model_type.arm, stream_type.audio)
        with open(audio_model_filepath, 'rb') as audio_model_file:
            dict_ar_models['audio'] = pickle.load(audio_model_file)

    return dict_ar_models


def get_model_filepath(str_model_rootdir, model_type_val, stream_type_val=None):
    
    str_model_dir = get_model_dir(str_model_rootdir, model_type_val)
    str_model_path = os.path.join(str_model_dir, stream_type_val.name + ".pkl")

    return str_model_path


def get_stream_type(str_stream_type):
    stype = None

    if str_stream_type == "rgb":
        stype = stream_type.rgb
    elif str_stream_type == "flow":
        stype = stream_type.flow
    elif str_stream_type == "audio":
        stype = stream_type.audio
    elif str_stream_type == "rgbflow":
        stype = stream_type.rgbflow
    elif str_stream_type == "rgbaudio":
        stype = stream_type.rgbaudio
    elif str_stream_type == "flowaudio":
        stype = stream_type.flowaudio
    elif str_stream_type == "rgbflowaudio":
        stype = stream_type.rgbflowaudio
    else:
        raise ValueError("Invalid stream type selection")

    return stype


def get_stream_type_string(stype):
    str_stream_type = None

    if stype == stream_type.rgb:
        str_stream_type = "rgb"
    elif stype == stream_type.flow:
        str_stream_type = "flow"
    elif stype == stream_type.audio:
        str_stream_type = "audio"
    elif stype == stream_type.rgbflow:
        str_stream_type = "rgbflow"
    elif stype == stream_type.rgbaudio:
        str_stream_type = "rgbaudio"
    elif stype == stream_type.flowaudio:
        str_stream_type = "flowaudio"
    elif stype == stream_type.rgbflowaudio:
        str_stream_type = "rgbflowaudio"
    else:
        raise ValueError("Invalid stream type selection")

    return str_stream_type


def get_stream_type_string_array(stype):

    str_first_modality = str_second_modality = str_third_modality = None

    if stype == stream_type.rgbflow or stype == stream_type.rgbaudio or stype == stream_type.rgbflowaudio:
        str_first_modality = "rgb"
        if stype == stream_type.rgbflow or stype == stream_type.rgbflowaudio:
            str_second_modality = "flow"
            if stype == stream_type.rgbflowaudio:
                str_third_modality = "audio"
        elif stype == stream_type.rgbaudio:
            str_second_modality = "audio"
    elif stype == stream_type.flowaudio:
        str_first_modality = "flow"
        str_second_modality = "audio"

    return str_first_modality, str_second_modality, str_third_modality


def get_split_files(str_folder, str_modality, split_id):

    train_setting_file = "train_" + str_modality + "_split%d.txt" % (split_id)
    train_split_file = os.path.join(str_folder, train_setting_file)
    val_setting_file = "val_" + str_modality + "_split%d.txt" % (split_id)
    val_split_file = os.path.join(str_folder, val_setting_file)
    test_setting_file = "test_" + str_modality + "_split%d.txt" % (split_id)
    test_split_file = os.path.join(str_folder, test_setting_file)

    if not os.path.exists(train_split_file) or not os.path.exists(val_split_file) or not os.path.exists(test_split_file):
        print("No split file exists in %s directory. Preprocess the dataset first" % (str_folder))

    return train_split_file, val_split_file, test_split_file


def parse_classnames(class_index_filepath):

    class_ind = [x.strip().split() for x in open(class_index_filepath)]
    class_names = [x[1] for x in class_ind]

    return class_names, class_ind


def count_model_parameters(model):

    # for name, p in model.named_parameters():
    #     if p.requires_grad:
    #         print("{} requires grad".format(name))
    #     else:
    #         print("{} no grad".format(name))

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def append_dict_values(dict_obj):

    dict_values = None

    for dict_key, values in dict_obj.items():
        if dict_values is None:
            dict_values = values
        else:
            tmp_dict_values = dict_values
            dict_values = np.append(tmp_dict_values, values, axis=0)

    return dict_values
