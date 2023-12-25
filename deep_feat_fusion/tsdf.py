from utils.utility_functions import *
from utils.feat_functions import *
from utils.data_structures import *

class TSDF:
    """Create a Stream Weighing Model Object 

    Args:
        str_config: The path of the configuration file
    """

    def __init__(self, str_modality, str_model_rootdir, str_result_rootdir, str_feat_rootdir, str_attention_mode):
        super(TSDF, self).__init__()

        if str_attention_mode == "swm":
            attention_mode = eFusion_Weight_Mode.swm
        elif str_attention_mode == "dwm":
            attention_mode = eFusion_Weight_Mode.dwm
        elif str_attention_mode == "swm_dwm":
            attention_mode = eFusion_Weight_Mode.swm_dwm
        else:
            raise ValueError("Invalid attention mode!")
        
        self.ar_modality = get_stream_type(str_modality)
        self.feat_rootdir = str_feat_rootdir
        self.str_result_rootdir = str_result_rootdir

        class_index_file = os.path.join("data", "class_index.txt")
        self.class_name_list, self.class_idx = parse_classnames(class_index_file)

        self.dict_ar_models = load_ar_models(str_model_rootdir, self.ar_modality)
        self.dict_ar_feat_scalers = load_ar_feat_scalers(str_feat_rootdir, self.ar_modality)

        # Initialize SWM and DWM models
        if attention_mode == eFusion_Weight_Mode.swm or attention_mode == eFusion_Weight_Mode.swm_dwm:
            swm_model_filepath = get_model_filepath(str_model_rootdir, model_type.swm, self.ar_modality)
            with open(swm_model_filepath, 'rb') as swm_model_file:
                self.swm_model = pickle.load(swm_model_file)

        if attention_mode == eFusion_Weight_Mode.dwm or attention_mode == eFusion_Weight_Mode.swm_dwm: 
            dwm_model_filepath = get_model_filepath(str_model_rootdir, model_type.dwm, self.ar_modality)
            with open(dwm_model_filepath, 'rb') as dwm_model_file:
                self.dwm_model = pickle.load(dwm_model_file)


    def train(self):

        print("training...")
        print("will be implemented later...")


    def get_ar_feat_targets(self, process_phase_val):
        print("getting activity recognition model features and their corresponding targets...")

        dict_ar_feats, dict_ar_labels = read_vid_features(self.feat_rootdir, self.ar_modality, process_phase_val)
        
        return dict_ar_feats, dict_ar_labels


    def get_swm_feat_targets(self, process_phase_val):

        print("getting stream weighing model features and their corresponding targets...")

        if self.ar_modality == stream_type.rgbflow:
            scaler_type_val = scaler_type.maxabsscale
            feat_fusion_type_val = feat_fusion_type.feat_add

        # Get the fused features at first
        dict_swm_feats = get_fused_feats(self.feat_rootdir, self.ar_modality, feat_fusion_type_val, process_phase_val)
        # Get the corresponding scaler for the fused features
        scaler_fused_feats = get_fused_scaler(self.feat_rootdir, self.ar_modality, feat_fusion_type_val, scaler_type_val)
        # Get the scaled features
        dict_swm_feats = scale_features(dict_swm_feats, scaler_fused_feats)
        # Get stream weighing target by using AR model probabilities
        dict_swm_targets = self.get_swm_targets(process_phase_val, "normalized")
            
        return dict_swm_feats, dict_swm_targets


    def get_dwm_feat_targets(self, process_phase_val):

        print("getting segment weighing model features and their corresponding targets...")

        if self.ar_modality == stream_type.rgbflow:
            scaler_type_val = scaler_type.maxabsscale
            feat_fusion_type_val = feat_fusion_type.append

        # Get the fused features at first
        dict_fused_feats = get_fused_feats(self.feat_rootdir, self.ar_modality, feat_fusion_type_val, process_phase_val)
        # Get the corresponding scaler for the fused features
        scaler_fused_feats = get_fused_scaler(self.feat_rootdir, self.ar_modality, feat_fusion_type_val, scaler_type_val)
        # Get the scaled features
        dict_fused_feats = scale_features(dict_fused_feats, scaler_fused_feats)
        # Get segment weighing target by using AR model probabilities
        dict_dwm_feats, dict_dwm_targets = self.get_dwm_targets(dict_fused_feats, process_phase_val)

        return dict_dwm_feats, dict_dwm_targets


    def predict_swm_weights(self, swm_feats, vid_name, dict_ar_feats, dict_ar_labels):

        if self.swm_model is None:
            raise ValueError("SWM model should be initialized or trained before prediction!")

        str_first_modality, str_second_modality, str_third_modality = get_stream_type_string_array(self.ar_modality)
        
        prob_first = self.dict_ar_models[str_first_modality].predict_proba(dict_ar_feats[str_first_modality][vid_name])
        prob_second = self.dict_ar_models[str_second_modality].predict_proba(dict_ar_feats[str_second_modality][vid_name])
        if str_third_modality is not None:
            prob_third = self.dict_ar_models[str_third_modality].predict_proba(dict_ar_feats[str_third_modality][vid_name])

        prob_gt_class_first = prob_first[:, dict_ar_labels[vid_name]]
        prob_gt_class_second = prob_second[:, dict_ar_labels[vid_name]]
        if str_third_modality is not None:
            prob_gt_class_third = prob_third[:, dict_ar_labels[vid_name]]
        
        if str_third_modality is not None:
            prob_gt_class = np.vstack((prob_gt_class_first, prob_gt_class_second, prob_gt_class_third))
        else:
            prob_gt_class = np.vstack((prob_gt_class_first, prob_gt_class_second))

        # compute the stream attention weights
        class_conf_sum = np.sum(prob_gt_class, axis=0) 
        swm_weights_ideal = prob_gt_class / class_conf_sum
        swm_weights_ideal = np.transpose(swm_weights_ideal)

        if str_third_modality is not None:
            swm_weights_equal = 0.5*np.ones(shape=(dict_ar_feats[str_first_modality][vid_name].shape[0], 3))
        else:
            swm_weights_equal = 0.5*np.ones(shape=(dict_ar_feats[str_first_modality][vid_name].shape[0], 2))

        swm_weights_preds = self.swm_model.predict(swm_feats)

        return swm_weights_preds, swm_weights_equal, swm_weights_ideal


    def predict_dwm_weights(self, dwm_feats, vid_name, dict_ar_feats, dict_ar_labels):

        if self.dwm_model is None:
            raise ValueError("DWM model should be initialized or trained before prediction!")

        str_first_modality, str_second_modality, str_third_modality = get_stream_type_string_array(self.ar_modality)        
        prob_first = self.dict_ar_models[str_first_modality].predict_proba(dict_ar_feats[str_first_modality][vid_name])
        prob_second = self.dict_ar_models[str_second_modality].predict_proba(dict_ar_feats[str_second_modality][vid_name])
        prob_final = prob_first + prob_second
        if str_third_modality is not None:
            prob_third = self.dict_ar_models[str_third_modality].predict_proba(dict_ar_feats[str_third_modality][vid_name])
            prob_final += prob_third

        segment_preds = np.argmax(prob_final, axis=1).tolist()

        dwm_weights_equal = np.ones((len(prob_final), ))
        dwm_weights_ideal = prob_final[:, dict_ar_labels[vid_name]]

        class_name = self.class_name_list[dict_ar_labels[vid_name]]
        # print(len(dwm_feats))
        dwm_weights_preds = np.zeros((len(dwm_feats), ))
        for segment_idx, segment_dwm_feat in enumerate(dwm_feats):
            segment_class_pred = segment_preds[segment_idx]
            segment_class_name = self.class_name_list[segment_class_pred]
            # print("segment_dwm_feat.shape = ", segment_dwm_feat.reshape(1, -1).shape)    
            dwm_weights_preds[segment_idx] = self.dwm_model[segment_class_name].predict(segment_dwm_feat.reshape(1, -1))
            # print("vidname/vidlabel_gt/seglabel_pred/weight_pred/weight_ideal = {} / {} / {} / {:.2f} / {:.2f}".format(vid_name, dict_ar_labels[vid_name], segment_class_pred, dwm_weights_preds[segment_idx], dwm_weights_ideal[segment_idx]))
        
        return dwm_weights_preds, dwm_weights_equal, dwm_weights_ideal


    def get_swm_targets(self, process_phase_val, swm_target_mode):

        dict_swm_targets = dict()

        dict_ar_feats, dict_ar_labels = read_vid_features(self.feat_rootdir, self.ar_modality, process_phase_val)
        str_first_modality, str_second_modality, str_third_modality = get_stream_type_string_array(self.ar_modality)
         
        # Weights from the class samples
        for idx, (vid_name, _) in enumerate(dict_ar_feats[str_first_modality].items()):

            # print(vid_name)
            
            prob_first = self.dict_ar_models[str_first_modality].predict_proba(dict_ar_feats[str_first_modality][vid_name])
            prob_gt_class_first = prob_first[:, dict_ar_labels[vid_name]]
            # print("prob gt first = ", prob_gt_class_first)

            prob_second = self.dict_ar_models[str_second_modality].predict_proba(dict_ar_feats[str_second_modality][vid_name])        
            prob_gt_class_second = prob_second[:, dict_ar_labels[vid_name]]
            # print("prob gt second = ", prob_gt_class_second)

            if str_third_modality is not None:
                # Third modality should be audio and patch-based processing is performed for the audio stream.
                # Thus, the segment probabilities of the true classes will be computed by taking the avearage over patch decisions
                prob_third = self.dict_ar_models[str_third_modality].predict_proba(dict_ar_feats[str_second_modality][vid_name])        
                prob_gt_class_third = prob_third[:, dict_ar_labels[vid_name]]

            if str_third_modality is not None:
                tmp = np.vstack((prob_gt_class_first, prob_gt_class_second))
                prob_gt_class = np.vstack((tmp, prob_gt_class_third))
            else:
                prob_gt_class = np.vstack((prob_gt_class_first, prob_gt_class_second))

            # compute the stream attention weights
            if swm_target_mode == "normalized":
                class_conf_sum = np.sum(prob_gt_class, axis=0)
                # Normalized ARM confidence values through models 
                stream_weights = np.transpose(prob_gt_class / class_conf_sum)
            elif swm_target_mode == "raw":
                # Raw ARM confidence values for the gt class 
                stream_weights = np.transpose(prob_gt_class)
            else:
                raise ValueError("Invalid selection of the mode of SWM training. Select normalized or raw values to train SWM model.")

            # Append feats
            dict_swm_targets[vid_name] = stream_weights
            # print("SWM weights for {} = {}".format(vid_name, stream_weights))

        return dict_swm_targets


    def get_dwm_targets(self, dict_fused_feats, process_phase_val):

        dict_dwm_feats = dict()
        dict_dwm_targets = dict()
        dict_dwm_feats_class = dict()
        dict_dwm_targets_class = dict()

        # Fuse features before training the weight models
        dict_ar_feats, dict_ar_labels = read_vid_features(self.feat_rootdir, self.ar_modality, process_phase_val)
        str_first_modality, str_second_modality, str_third_modality = get_stream_type_string_array(self.ar_modality)

        # Weights from the class samples
        for idx, (vid_name, vid_fused_feats) in enumerate(dict_fused_feats.items()):

            # print(vid_name)
            
            prob_first = self.dict_ar_models[str_first_modality].predict_proba(dict_ar_feats[str_first_modality][vid_name])
            prob_gt_class_first = prob_first[:, dict_ar_labels[vid_name]]
            # print("prob gt first = ", prob_gt_class_first)

            prob_second = self.dict_ar_models[str_second_modality].predict_proba(dict_ar_feats[str_second_modality][vid_name])        
            prob_gt_class_second = prob_second[:, dict_ar_labels[vid_name]]
            # print("prob gt second = ", prob_gt_class_second)
            prob_pred_sum = prob_first + prob_second
            prob_gt_class = np.vstack((prob_gt_class_first, prob_gt_class_second))
            if str_third_modality is not None:
                # Third modality should be audio and patch-based processing is performed for the audio stream.
                # Thus, the segment probabilities of the true classes will be computed by taking the avearage over patch decisions
                prob_third = self.dict_ar_models[str_third_modality].predict_proba(dict_ar_feats[str_second_modality][vid_name])        
                prob_gt_class_third = prob_third[:, dict_ar_labels[vid_name]]
                prob_pred_sum += prob_third
                prob_gt_class = np.vstack((prob_gt_class, prob_gt_class_third))
            # compute the stream attention weights
            class_conf_sum = np.sum(prob_gt_class, axis=0) 

            if process_phase_val == process_phase.train:
                # Define the related key value 
                class_name_key = self.class_name_list[dict_ar_labels[vid_name]] # select using predictions

                if class_name_key in dict_dwm_feats_class.keys():
                    tmp = dict_dwm_feats_class[class_name_key]
                    dict_dwm_feats_class[class_name_key] = np.append(tmp, vid_fused_feats, axis=0)
                else:
                    dict_dwm_feats_class[class_name_key] = vid_fused_feats

                # append targets
                if class_name_key in dict_dwm_targets_class.keys():
                    tmp = dict_dwm_targets_class[class_name_key]
                    dict_dwm_targets_class[class_name_key] = np.append(tmp, class_conf_sum, axis=0)
                else:
                    dict_dwm_targets_class[class_name_key] = class_conf_sum
                print("DWM weights for {} = {}".format(vid_name, class_conf_sum))

            elif process_phase_val == process_phase.test:
                dict_dwm_feats[vid_name] = vid_fused_feats
                dict_dwm_targets[vid_name] = class_conf_sum
            else:
                raise ValueError("Invalid DWM extraction mode")

        if process_phase_val == process_phase.train:
            # # Populate with the samples of other classes (set to zero weights)
            for idx_ref, (class_ref, vid_feats_ref) in enumerate(dict_dwm_feats_class.items()):
                print("reference action = ", class_ref)
                nof_class = len(self.class_name_list)
                nof_samples_each_class = int(np.ceil(vid_feats_ref.shape[0]/(nof_class-1)))
                dwm_feats_other = None
                for idx_tmp, (class_other, vid_feats_other) in enumerate(dict_dwm_feats_class.items()):
                    # If the class indexes are from the same activity, continue
                    if class_other == class_ref:
                        continue
                    print("action name for negative samples = ", class_other)
                    nof_total_feats = vid_feats_other.shape[0]
                    if nof_samples_each_class > nof_total_feats: # If the number of samples are not enough, take all samples
                        nof_random_samples = nof_total_feats    
                    else:
                        nof_random_samples = nof_samples_each_class    

                    rand_idx_list = np.random.choice(nof_total_feats, size=nof_random_samples, replace=False)
                    if dwm_feats_other is None:
                        dwm_feats_other = vid_feats_other[rand_idx_list, :]
                    else:
                        dwm_feats_other = np.append(dwm_feats_other, vid_feats_other[rand_idx_list, :], axis=0) 
                
                print("class samples size for action {} = {}".format(class_ref, dict_dwm_feats_class[class_ref].shape))
                print("other samples size for action {} = {}".format(class_ref, dwm_feats_other.shape))
                dict_dwm_feats[class_ref] = np.append(dict_dwm_feats_class[class_ref], dwm_feats_other, axis=0)
                print("appended sample size for {} = {}".format(class_ref, dict_dwm_feats[class_ref].shape))
                print("*****")
                dict_dwm_targets[class_ref] = np.append(dict_dwm_targets_class[class_ref], np.zeros((dwm_feats_other.shape[0],)), axis=0)

        return dict_dwm_feats, dict_dwm_targets


    def get_decisions_using_weights(self, swm_weights, dwm_weights, vid_name, dict_ar_feats, vid_label):

        prob_first = prob_second = prob_third = None
        str_first_modality, str_second_modality, str_third_modality = get_stream_type_string_array(self.ar_modality)
        prob_first = self.dict_ar_models[str_first_modality].predict_proba(dict_ar_feats[str_first_modality][vid_name])
        prob_second = self.dict_ar_models[str_second_modality].predict_proba(dict_ar_feats[str_second_modality][vid_name])
        if str_third_modality is not None:
            prob_third = self.dict_ar_models[str_third_modality].predict_proba(dict_ar_feats[str_third_modality][vid_name])

        nof_class = len(self.class_name_list)

        first_w = swm_weights[:, 0]
        second_w = swm_weights[:, 1]

        if prob_third is None:
            prob_final = prob_first + prob_second
            weighted_probs = np.multiply(prob_first, first_w[:, np.newaxis]) + np.multiply(prob_second, second_w[:, np.newaxis])
        else:
            third_w = swm_weights[:, 2]
            weighted_probs = np.multiply(prob_first, first_w[:, np.newaxis]) + np.multiply(prob_second, second_w[:, np.newaxis]) + np.multiply(prob_third, third_w[:, np.newaxis])
            prob_final = prob_first + prob_second + prob_third
        
        segment_preds = np.argmax(weighted_probs, axis=1).tolist()
        decision_weights = prob_final[:, vid_label] 
        # print("segment weights = ", dwm_weights)
        model_class_pred_sums = np.zeros((nof_class, ))
        model_class_prob = np.zeros((nof_class, ))
        for class_idx in np.arange(nof_class):
            seg_idx_list = np.where(segment_preds == class_idx)[0].tolist()
            model_class_pred_sums[class_idx] = len(seg_idx_list)
            for seg_idx in seg_idx_list:
                model_class_prob[class_idx] += dwm_weights[seg_idx] 

        vid_label_pred_prob = np.argmax(model_class_prob)
        vid_label_pred_majority = np.argmax(model_class_pred_sums)

        return segment_preds, vid_label_pred_prob, vid_label_pred_majority