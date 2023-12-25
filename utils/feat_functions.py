import os
import pickle
from collections import defaultdict
from sklearn import preprocessing
from utils.data_structures import *


def get_feat_dir(feat_rootdir, feat_type_val):
    
    if feat_type_val == feat_type.single:
        str_feat_dir = os.path.join(feat_rootdir, "single_feats")
    elif feat_type_val == feat_type.fused:
        str_feat_dir = os.path.join(feat_rootdir, "fused_feats")
    else:
        raise ValueError("Undefined feat type!")

    return str_feat_dir


def get_scaler_filepath(feat_rootdir, feat_type_val, stream_type_val, feat_fusion_type_val=None, scaler_type_val=None):
    
    str_feat_dir = get_feat_dir(feat_rootdir, feat_type_val)

    if feat_type_val == feat_type.single:
        str_scaler_filepath = os.path.join(str_feat_dir, stream_type_val.name + "_feats_scaler.pkl")
    elif feat_type_val == feat_type.fused:
        str_scaler_filepath = os.path.join(str_feat_dir, stream_type_val.name + "_feats_" + feat_fusion_type_val.name + "_" + scaler_type_val.name + "_scaler.pkl")
    else:
        raise ValueError("Undefined feat type for scaler!")

    return str_scaler_filepath


def get_feat_filepath(feat_rootdir, feat_type_val, feat_fusion_type_val, stream_type_val, process_phase_val):
    
    str_feat_dir = get_feat_dir(feat_rootdir, feat_type_val)

    if feat_type_val == feat_type.single:
        str_scaler_filepath = os.path.join(str_feat_dir, stream_type_val.name + "_feats_" + process_phase_val.name + ".pkl")
    elif feat_type_val == feat_type.fused:
        str_scaler_filepath = os.path.join(str_feat_dir, stream_type_val.name + "_feats_" + feat_fusion_type_val.name + "_" + process_phase_val.name + ".pkl")
    else:
        raise ValueError("Undefined feat type for scaler!")

    return str_scaler_filepath


def get_fused_feats(feat_rootdir, stream_type_val, feat_fusion_type_val, process_phase_val):

    fused_feats_filepath = get_feat_filepath(feat_rootdir, feat_type.fused, feat_fusion_type_val, stream_type_val, process_phase.test)
    if os.path.isfile(fused_feats_filepath):
        with open(fused_feats_filepath, 'rb') as feat_file:
            dict_fused_feats = pickle.load(feat_file)
    else:
        # Read test features for SWM
        dict_ar_feats, _ = read_vid_features(feat_rootdir, stream_type_val, process_phase_val)
        dict_fused_feats = fuse_features(stream_type_val, feat_fusion_type_val, feat_fusion_level.video, dict_ar_feats)
        # Save features which are not scaled (this data will be used with other scalers after)
        with open(fused_feats_filepath, 'wb') as feat_file:
            pickle.dump(dict_fused_feats, feat_file, pickle.HIGHEST_PROTOCOL)

    return dict_fused_feats


def get_fused_scaler(feat_rootdir, stream_type_val, feat_fusion_type_val, scaler_type_val):

    fused_scaler_filepath = get_scaler_filepath(feat_rootdir, feat_type.fused, stream_type_val, feat_fusion_type_val, scaler_type_val)
    if os.path.isfile(fused_scaler_filepath):
        with open(fused_scaler_filepath, 'rb') as scaler_file:
            fused_scaler = pickle.load(scaler_file)
    else:
        # Read test features for SWM
        dict_ar_feats, _ = read_vid_features(feat_rootdir, stream_type_val, process_phase.train)
        dict_fused_feats = fuse_features(stream_type_val, feat_fusion_type_val, feat_fusion_level.video, dict_ar_feats)
        # Get scaler and save features
        fused_scaler = get_scaler(dict_fused_feats, scaler_type_val)
        # Save features which are not scaled (this data will be used with other scalers after)
        with open(fused_scaler_filepath, 'wb') as scaler_file:
            pickle.dump(fused_scaler, scaler_file, pickle.HIGHEST_PROTOCOL)

    return fused_scaler


def load_ar_feat_scalers(feat_rootdir, stream_type_val):

    dict_ar_feat_scalers = defaultdict(dict)

    if stream_type_val == stream_type.rgbflow or stream_type_val == stream_type.rgbaudio or stream_type_val == stream_type.rgbflowaudio:
        rgb_scaler_filepath = get_scaler_filepath(feat_rootdir, feat_type.single, stream_type.rgb)
        with open(rgb_scaler_filepath, 'rb') as rgb_scaler_file:
            dict_ar_feat_scalers['rgb'] = pickle.load(rgb_scaler_file)

    if stream_type_val == stream_type.rgbflow or stream_type_val == stream_type.flowaudio or stream_type_val == stream_type.rgbflowaudio:
        flow_scaler_filepath = get_scaler_filepath(feat_rootdir, feat_type.single, stream_type.flow)
        with open(flow_scaler_filepath, 'rb') as flow_scaler_file:
            dict_ar_feat_scalers['flow'] = pickle.load(flow_scaler_file)

    if stream_type_val == stream_type.rgbaudio or stream_type_val == stream_type.flowaudio or stream_type_val == stream_type.rgbflowaudio:
        get_scaler_filepath(feat_rootdir, feat_type.single, stream_type.audio)
        with open(audio_scaler_filepath, 'rb') as audio_scaler_file:
            dict_ar_feat_scalers['audio'] = pickle.load(audio_scaler_file)

    return dict_ar_feat_scalers


def read_vid_features(feat_rootdir, stream_type_val, process_phase_val):

    print("reading ARN {} features...".format(process_phase_val.name))
    dict_arn_feats = defaultdict(dict)
    dict_arn_labels = None

    if stream_type_val == stream_type.rgbflow or stream_type_val == stream_type.rgbaudio or stream_type_val == stream_type.rgbflowaudio:
        rgb_feat_dir = get_feat_dir(feat_rootdir, feat_type.single)
        dict_arn_feats['rgb'], dict_arn_labels = read_single_feats(process_phase_val, stream_type.rgb, rgb_feat_dir)

    if stream_type_val == stream_type.rgbflow or stream_type_val == stream_type.flowaudio or stream_type_val == stream_type.rgbflowaudio:
        flow_feat_dir = get_feat_dir(feat_rootdir, feat_type.single)
        dict_arn_feats['flow'], dict_arn_tmp = read_single_feats(process_phase_val, stream_type.flow, flow_feat_dir)
        if dict_arn_labels is None:
            dict_arn_labels = dict_arn_tmp
        elif (dict_arn_labels != dict_arn_tmp):
            raise ValueError("Labels are not compatible!")

    if stream_type_val == stream_type.rgbaudio or stream_type_val == stream_type.flowaudio or stream_type_val == stream_type.rgbflowaudio:
        audio_feat_dir = get_feat_dir(feat_rootdir, feat_type.single)
        dict_arn_feats['audio'], dict_arn_tmp = read_single_feats(process_phase_val, stream_type.audio, audio_feat_dir)
        if dict_arn_labels is None:
            dict_arn_labels = dict_arn_tmp
        elif (dict_arn_labels != dict_arn_tmp):
            raise ValueError("Labels are not compatible!")

    return dict_arn_feats, dict_arn_labels


def fuse_features(modality, feat_fusion_type_val, feat_fusion_level_val, dict_arn_feats):

    feats_first = feats_second = feats_third = None

    if feat_fusion_level_val != feat_fusion_level.video and feat_fusion_level_val != feat_fusion_level.segment:
        raise ValueError("Invalid selection of feature level. It should be segment or vid!")

    if feat_fusion_type_val != feat_fusion_type.append and \
        feat_fusion_type_val != feat_fusion_type.feat_max and \
        feat_fusion_type_val != feat_fusion_type.feat_mean and \
        feat_fusion_type_val != feat_fusion_type.feat_add and \
        feat_fusion_type_val != feat_fusion_type.cbp:
        raise ValueError("Invalid selection of feature fusion. It should be append, feat_max, feat_mean, feat_add or cbp!")

    if modality == stream_type.rgbflow or modality == stream_type.rgbaudio or modality == stream_type.rgbflowaudio:
        feats_first = dict_arn_feats['rgb']

    if modality == stream_type.rgbflow or modality == stream_type.flowaudio or modality == stream_type.rgbflowaudio:
        if modality == stream_type.flowaudio:
            feats_first = dict_arn_feats['flow']
        else:
            feats_second = dict_arn_feats['flow']

    if modality == stream_type.rgbaudio or modality == stream_type.flowaudio or modality == stream_type.rgbflowaudio:
        if modality == stream_type.rgbflowaudio:
            feats_third = dict_arn_feats['audio']
        else:
            feats_second = dict_arn_feats['audio']

    if feat_fusion_level_val == feat_fusion_level.segment:

        if feat_fusion_type_val == feat_fusion_type.feat_max:
            if feats_third is None:
                feats_fused = np.maximum(np.array(feats_first, dtype=int), np.array(feats_second, dtype=int)).tolist()
            else:
                feats_fused = np.maximum(np.array(feats_first, dtype=int), np.array(feats_second, dtype=int), np.array(feats_third, dtype=int)).tolist()
        elif feat_fusion_type_val == feat_fusion_type.feat_mean:
            if feats_third is None:
                feats_fused = ((np.array(feats_first, dtype=np.float) + np.array(feats_second, dtype=np.float) + np.array(feats_third, dtype=np.float))/3.0).tolist()
            else:
                feats_fused = ((np.array(feats_first, dtype=np.float) + np.array(feats_second, dtype=np.float))/2.0).tolist()
        elif feat_fusion_type_val == feat_fusion_type.append:
            if feats_third is None:
                feats_fused = np.append(feats_first, feats_second, axis=1)
            else:
                feats_fused = np.append(np.append(feats_first, feats_second, axis=1), feats_third, axis=1)
        elif feat_fusion_type_val == feat_fusion_type.feat_add:
            if feats_third is None:
                feats_fused = feats_first + feats_second
            else:
                feats_fused = feats_first + feats_second + feats_third
        elif feat_fusion_type_val == feat_fusion_type.cbp:
            if feats_third is None:
                feats_fused = mcb.mcb(feats_first, feats_second, d=2048)
            else:
                raise ValueError("Not implemented!")
        else:
            raise ValueError("Invalid selection of feature fusion type!")
    
    else:

        feats_fused = dict()
        feats_vid_all = None
        for vid_name, vid_feats_first in feats_first.items():       
            vid_feats_second = feats_second[vid_name]
            if feats_third is not None:
                vid_feats_third = feats_third[vid_name]
            if feat_fusion_type_val == feat_fusion_type.feat_max:
                if feats_third is not None:
                    feats_vid_tmp = np.maximum(np.array(vid_feats_first, dtype=int), np.array(vid_feats_second, dtype=int), np.array(vid_feats_third, dtype=int)).tolist()
                else:
                    feats_vid_tmp = np.maximum(np.array(vid_feats_first, dtype=int), np.array(vid_feats_second, dtype=int)).tolist()
            elif feat_fusion_type_val == feat_fusion_type.feat_mean:
                if feats_third is not None:
                    feats_vid_tmp = ((np.array(vid_feats_first, dtype=np.float) + np.array(vid_feats_second, dtype=np.float) + np.array(vid_feats_third, dtype=np.float))/3.0).tolist()
                else:
                    feats_vid_tmp = ((np.array(vid_feats_first, dtype=np.float) + np.array(vid_feats_second, dtype=np.float))/2.0).tolist()
            elif feat_fusion_type_val == feat_fusion_type.append:
                if feats_third is None:
                    feats_vid_tmp = np.append(vid_feats_first, vid_feats_second, axis=1)
                else:
                    feats_vid_tmp = np.append(np.append(vid_feats_first, vid_feats_second, axis=1), vid_feats_third, axis=1)
            elif feat_fusion_type_val == feat_fusion_type.feat_add:
                if feats_third is None:
                    feats_vid_tmp = vid_feats_first + vid_feats_second
                else:
                    feats_vid_tmp = vid_feats_first + vid_feats_second + vid_feats_third
            elif feat_fusion_type_val == feat_fusion_type.cbp:
                if feats_third is None:
                    feats_vid_tmp = mcb.mcb(vid_feats_first, vid_feats_second, d=2048)
                else:
                    raise ValueError("Not implemented!")
            else:
                raise ValueError("Invalid selection of feature fusion type!")

            feats_fused[vid_name] = feats_vid_tmp
            
    return feats_fused


def get_scaler(dict_feats, scaler_type):

    # This is training phase. Thus, all features will be appended and scaler will be computed
    scaler = get_scaler_base(scaler_type)

    # Append all training features at first 
    feats_vid_all = None
    for vid_name, feats_vid_tmp in dict_feats.items():
        if feats_vid_all is None:
            feats_vid_all = feats_vid_tmp
        else:
            feats_vid_all = np.append(feats_vid_all, feats_vid_tmp, axis=0)

    scaler = scaler.fit(feats_vid_all)

    return scaler


def scale_features(dict_feats, scaler):

    dict_feats_scaled = dict()

    # This part is common for train and test phases. Apply scaler to the features
    for vid_name, feats_vid_tmp in dict_feats.items():
        if vid_name in dict_feats_scaled.keys():
            raise ValueError("Feature scaling was performed before for {}!".format(vid_name))
        dict_feats_scaled[vid_name] = scaler.transform(feats_vid_tmp)

    return dict_feats_scaled


def get_scaler_base(scaler_type_val):

    scaler_obj = None

    if scaler_type_val == scaler_type.standardscaler:
        scaler_obj = preprocessing.StandardScaler()
    elif scaler_type_val == scaler_type.minmax:
        scaler_obj = preprocessing.MinMaxScaler()
    elif scaler_type_val == scaler_type.robustscaler:
        scaler_obj = preprocessing.RobustScaler()
    elif scaler_type_val == scaler_type.maxabsscale:
        scaler_obj = preprocessing.MaxAbsScaler()
    elif scaler_type_val == scaler_type.normalizer:
        scaler_obj = preprocessing.Normalizer()
    elif scaler_type_val == scaler_type.powertransformer:
        scaler_obj = preprocessing.PowerTransformer()
    else:
        raise ValueError("Invalid scaler selection!")

    return scaler_obj


def read_single_feats(process_phase_val, modality, str_feat_dir):

    # Check parameters
    if process_phase_val != process_phase.train and process_phase_val != process_phase.test:
        raise ValueError("Invalid selection of feature mode. It should be train or test!")
    
    scaler_filepath = os.path.join(str_feat_dir, modality.name + "_feats_scaler.pkl")
    
    if process_phase_val == process_phase.train:

        feat_filepath_train = os.path.join(str_feat_dir, modality.name + "_feats_train.pkl")
        train_feats_vid, train_labels_vid = read_feat_file(feat_filepath_train, scaler_filepath)
        feat_filepath_val = os.path.join(str_feat_dir, modality.name + "_feats_val.pkl")
        val_feats_vid, val_labels_vid = read_feat_file(feat_filepath_val, scaler_filepath)
        print("Nof train videos = {}".format(len(train_labels_vid)))
        print("Nof validation videos = {}".format(len(val_labels_vid)))
        
        feats_single_modal = train_feats_vid
        feats_single_modal.update(val_feats_vid)
        labels_single_modal = train_labels_vid
        labels_single_modal.update(val_labels_vid)
            
    elif process_phase_val == process_phase.test:
        # Get only the features at video level for test samples
        feat_filepath_test = os.path.join(str_feat_dir, modality.name + "_feats_test.pkl")
        test_feats_vid, test_labels_vid = read_feat_file(feat_filepath_test, scaler_filepath)
        print("Nof test videos = {}".format(len(test_labels_vid)))
        feats_single_modal = test_feats_vid
        labels_single_modal = test_labels_vid

    return feats_single_modal, labels_single_modal


def read_feat_file(feat_filepath, scaler_filepath):

    # Read video samples

    with open(feat_filepath, 'rb') as feat_file:
        feats = pickle.load(feat_file)
        labels = pickle.load(feat_file)

    with open(scaler_filepath, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
        for k, v in feats.items():
            feats[k] = scaler.transform(v)

    return feats, labels
