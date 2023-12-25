import os
import argparse
import pickle
from configparser import ConfigParser
from collections import Counter
import itertools

from utils.data_structures import *
from utils.utility_functions import *
from utils.visualization import plot_confusion_matrix, plot_class_detections, plot_regression_diff
from deep_feat_fusion.tsdf import TSDF
# Third party module imports
from sklearn import svm, preprocessing

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, RepeatedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, mean_squared_error, mean_absolute_error

config_parser = ConfigParser()

param_parser = argparse.ArgumentParser(description='Test Two-Stage Decision Fusion Algorithm for Multi-modal Egocentric Activity Recognition')
param_parser.add_argument('--config', '-c', required=True, help='configuration file path')


def test_tsdf_fusion(str_config_path):

    # Parse configuration file at first
    config_parser.read(str_config_path)
    str_result_rootdir = config_parser.get('common_params', 'result_rootdir')
    str_feat_rootdir = config_parser.get('common_params', 'feat_rootdir')
    str_modality = config_parser.get('ar_common_params', 'modality')
    ar_modality = get_stream_type(str_modality)
    str_model_rootdir = config_parser.get('common_params', 'model_rootdir')
    str_attention_mode = config_parser.get('ar_attention_params', 'attention_mode')

    class_index_file = os.path.join("data", "class_index.txt")
    class_name_list, _ = parse_classnames(class_index_file)

    tsdf_obj = TSDF(str_modality, str_model_rootdir, str_result_rootdir, str_feat_rootdir, str_attention_mode)

    # Get features and targets at first
    dict_ar_test_feats, dict_ar_test_labels = tsdf_obj.get_ar_feat_targets(process_phase.test)
    dict_swm_feats_test, dict_swm_targets_test = tsdf_obj.get_swm_feat_targets(process_phase.test)
    dict_dwm_feats_test, dict_dwm_targets_test = tsdf_obj.get_dwm_feat_targets(process_phase.test)

    # Get model-based predictions for SWM/DWM and evaluate results
    mse_test = mae_test = std_test = None
    perf_prob = PerformanceStats(len(class_name_list))
    perf_majority = PerformanceStats(len(class_name_list))
    perf_prob_equal = PerformanceStats(len(class_name_list))
    perf_majority_equal = PerformanceStats(len(class_name_list))

    segvid_labels = SegVidLabels()
    segvid_labels_equal = SegVidLabels()

    swm_weights_preds_all = None

    print("getting model-based weighing results for each videos...")
    for vid_name, vid_label_gt in dict_ar_test_labels.items():    

        # Get stream weight predictions
        swm_weights_preds, swm_weights_equal, swm_weights_ideal = tsdf_obj.predict_swm_weights(dict_swm_feats_test[vid_name], vid_name, dict_ar_test_feats, dict_ar_test_labels)
        if swm_weights_preds_all is None:
            swm_weights_preds_all = swm_weights_preds
        else:
            swm_weights_preds_all = np.append(swm_weights_preds_all, swm_weights_preds, axis=0)
        # Get segment weight predictions
        dwm_weights_preds, dwm_weights_equal, dwm_weights_ideal = tsdf_obj.predict_dwm_weights(dict_dwm_feats_test[vid_name], vid_name, dict_ar_test_feats, dict_ar_test_labels)
        # Set segment labels
        nof_segments = swm_weights_preds.shape[0]
        segment_label_gt = (vid_label_gt*np.ones(nof_segments, dtype=int)).tolist()

        # Get decisions with using the given stream and segment weights
        seg_pred, \
        vid_pred_prob, \
        vid_pred_majority = tsdf_obj.get_decisions_using_weights(swm_weights_preds, dwm_weights_preds, vid_name, dict_ar_test_feats, vid_label_gt)
        segvid_labels.update(seg_pred, vid_pred_prob, vid_pred_majority, segment_label_gt, vid_label_gt)

        # Get decisions for equally weighted streams 
        seg_pred_equal, \
        vid_pred_prob_equal, \
        vid_pred_majority_equal = tsdf_obj.get_decisions_using_weights(swm_weights_equal, dwm_weights_equal, vid_name, dict_ar_test_feats, vid_label_gt)
        segvid_labels_equal.update(seg_pred_equal, vid_pred_prob_equal, vid_pred_majority_equal, segment_label_gt, vid_label_gt)

        # print(segment_label_gt)

        # Get prob-based decisions using regressor weights and update
        seg_transitions_prob, \
        vid_transitions_prob, \
        seg_hist, \
        vid_hist = get_transition_stats(class_name_list, segment_label_gt, vid_label_gt, seg_pred_equal, vid_pred_prob_equal, seg_pred, vid_pred_prob)
        perf_prob.update(seg_hist, vid_hist, seg_transitions_prob, vid_transitions_prob)

        # Get mv-based decisions using regressor weights and update
        seg_transitions_mv, \
        vid_transitions_mv, \
        seg_hist, \
        vid_hist = get_transition_stats(class_name_list, segment_label_gt, vid_label_gt, seg_pred_equal, vid_pred_majority_equal, seg_pred, vid_pred_majority)
        perf_majority.update(seg_hist, vid_hist, seg_transitions_mv, vid_transitions_mv)

        # Get prob-based decisions using equal weights and update
        seg_transitions_prob_equal, \
        vid_transitions_prob_equal, \
        seg_hist, \
        vid_hist = get_transition_stats(class_name_list, segment_label_gt, vid_label_gt, seg_pred_equal, vid_pred_prob_equal, seg_pred_equal, vid_pred_prob_equal)
        perf_prob_equal.update(seg_hist, vid_hist, seg_transitions_prob_equal, vid_transitions_prob_equal)

        # Get mv-based decisions using equal weights and update
        seg_transitions_prob_mv, \
        vid_transitions_prob_mv, \
        seg_hist, \
        vid_hist = get_transition_stats(class_name_list, segment_label_gt, vid_label_gt, seg_pred_equal, vid_pred_majority_equal, seg_pred_equal, vid_pred_majority_equal)
        perf_majority_equal.update(seg_hist, vid_hist, seg_transitions_prob_mv, vid_transitions_prob_mv)

    # Compare model-based predictions and targets for stream weighing
    swm_targets_test = append_dict_values(dict_swm_targets_test)
    mse_test, mae_test, std_test = plot_regression_diff("test",  ar_modality, swm_targets_test, swm_weights_preds_all)

    print("Evaluating probability-based performance for equal weighing...")
    eval_results_prob_equal = evaluate_performance(str_result_rootdir, class_name_list, segvid_labels_equal, "prob")
    print("Evaluating performance using majority voting for equal weighing...")
    eval_results_mv_equal = evaluate_performance(str_result_rootdir, class_name_list, segvid_labels_equal, "majority")
    # Validate segment decisions
    if (eval_results_prob_equal.get_seg_acc() != eval_results_mv_equal.get_seg_acc()).any() or \
        (eval_results_prob_equal.get_seg_f1score() != eval_results_mv_equal.get_seg_f1score()).any() or \
        (eval_results_prob_equal.get_seg_confmat() != eval_results_mv_equal.get_seg_confmat()).any():
        raise ValueError("Segment decision performances should be equal to each other for majority and probabilistic calculation")
    # Validate state transition for equally weighted streams. State transitions should be 0 for FT and TF
    if perf_prob_equal.get_segment_transitions()['TF'] > 0 or \
        perf_prob_equal.get_segment_transitions()['FT'] > 0 or \
        perf_majority_equal.get_segment_transitions()['TF'] > 0 or \
        perf_majority_equal.get_segment_transitions()['FT'] > 0:
        raise ValueError("Decision transitions should be zero from False to True or vice versa for equally weighted streams!")

    print("Evaluating probability-based performance for model-based weighing...")
    eval_results_prob = evaluate_performance(str_result_rootdir, class_name_list, segvid_labels, "prob")
    print("Evaluating performance using majority voting for model-based weighing...")
    eval_results_mv = evaluate_performance(str_result_rootdir, class_name_list, segvid_labels, "majority")
    # Validate segment decisions
    if (eval_results_prob.get_seg_acc() != eval_results_mv.get_seg_acc()).any() or \
        (eval_results_prob.get_seg_f1score() != eval_results_mv.get_seg_f1score()).any() or \
        (eval_results_prob.get_seg_confmat() != eval_results_mv.get_seg_confmat()).any():
        raise ValueError("Segment decision performances should be equal to each other for majority and probabilistic calculation")

    # Write evaluation results to the file
    write_eval_results(str_result_rootdir, class_name_list, mae_test, std_test, eval_results_prob, eval_results_mv, eval_results_prob_equal, eval_results_mv_equal, perf_prob, perf_majority, perf_prob_equal, perf_majority_equal)


def evaluate_performance(str_result_rootdir, class_name_list, segvid_labels, eval_mode):

    vid_label_gt = segvid_labels.get_vid_label_gt()
    segment_label_gt = segvid_labels.get_seg_label_gt() 
    segment_label_pred = segvid_labels.get_seg_label() 
    if eval_mode == "prob":
        vid_label_pred = segvid_labels.get_vid_label_prob() 
    elif eval_mode == "majority":
        vid_label_pred = segvid_labels.get_vid_label_mv()

    print("Total nof videos used for evaluation = {}".format(len(vid_label_gt)))
    print("Total nof video segments used for evaluation = {}".format(len(segment_label_gt)))

    vid_accuracy = accuracy_score(vid_label_gt, vid_label_pred)
    vid_f1 = f1_score(vid_label_gt, vid_label_pred, average='weighted')

    conf_mat_video = confusion_matrix(vid_label_gt, vid_label_pred)
    # Plot confusion matrices
    fig_savepath = os.path.join(str_result_rootdir, "vidbased_normalized_cm.png")
    plot_confusion_matrix(conf_mat_video, class_name_list, fig_savepath, normalize='True', title='Video-based Normalized Confusion Matrix')
    fig_savepath = os.path.join(str_result_rootdir, "vidbased_cm.png")
    plot_confusion_matrix(conf_mat_video, class_name_list, fig_savepath, normalize='False', title='Video-based Confusion Matrix')
    # Plot stacked bar for true and false detections
    true_detections_video = conf_mat_video.diagonal()
    false_detections_video = conf_mat_video.sum(axis=1) - true_detections_video 
    fig_savepath = os.path.join(str_result_rootdir, "vidbased_true_false_detections.png")
    plot_class_detections(true_detections_video, false_detections_video, class_name_list, fig_savepath)

    # Evaluate segment-based performance
    # test_feats_all, test_labels_all = get_segment_features(vid_features, vid_labels)
    segment_accuracy = accuracy_score(segment_label_gt, segment_label_pred)
    segment_f1 = f1_score(segment_label_gt, segment_label_pred, average='weighted')

    conf_mat_segment = confusion_matrix(segment_label_gt, segment_label_pred)
    fig_savepath = os.path.join(str_result_rootdir, "segbased_normalized_cm.png")
    plot_confusion_matrix(conf_mat_segment, class_name_list, fig_savepath, normalize='True', title='Segment-based Normalized Confusion Matrix')
    fig_savepath = os.path.join(str_result_rootdir, "segbased_cm.png")
    plot_confusion_matrix(conf_mat_segment, class_name_list, fig_savepath, normalize='False', title='Segment-based Confusion Matrix')
    # Plot stacked bar for true and false detections
    true_detections_seg = conf_mat_segment.diagonal()
    false_detections_seg = conf_mat_segment.sum(axis=1) - true_detections_seg 
    fig_savepath = os.path.join(str_result_rootdir, "segbased_true_false_detections.png")
    plot_class_detections(true_detections_seg, false_detections_seg, class_name_list, fig_savepath)

    # Normalize confusion matrix
    conf_mat_video = conf_mat_video.astype('float') / conf_mat_video.sum(axis=1)[:, np.newaxis]
    conf_mat_segment = conf_mat_segment.astype('float') / conf_mat_segment.sum(axis=1)[:, np.newaxis]

    # Plot class-based accuracies
    class_acc_vid = conf_mat_video.diagonal()
    class_acc_seg = conf_mat_segment.diagonal()

    return EvaluationResults(vid_accuracy, vid_f1, segment_accuracy, segment_f1, conf_mat_video, conf_mat_segment)


def get_transition_stats(class_name_list, segment_gt, vid_label_gt, segment_pred_equal, vid_pred_equal, segment_pred, vid_pred):

    dict_segment_stats = {'TT':0, 'TF':0, 'FT':0, 'FF':0}
    segment_hist = np.zeros((len(class_name_list),4), dtype=np.int32)
    dict_vid_stats = {'TT':0, 'TF':0, 'FT':0, 'FF':0}
    vid_hist = np.zeros((len(class_name_list),4), dtype=np.int32)

    # print("segment gt size = ", len(segment_gt))
    for idx, segment_label in enumerate(segment_gt):
        # print("idx / segment_label = {}/{}".format(idx, segment_label))
        pred_equal = segment_pred_equal[idx]
        pred_weighted = segment_pred[idx]
        if segment_label == pred_equal:
            if segment_label == pred_weighted:
                dict_segment_stats['TT'] += 1
                segment_hist[segment_label][0]+=1
            else:
                dict_segment_stats['TF'] += 1
                segment_hist[segment_label][1]+=1
        else:
            if segment_label == pred_weighted:
                dict_segment_stats['FT'] += 1
                segment_hist[segment_label][2]+=1
            else:
                dict_segment_stats['FF'] += 1
                segment_hist[segment_label][3]+=1

    if vid_label_gt == vid_pred_equal:
        if vid_label_gt == vid_pred:
            dict_vid_stats['TT'] += 1
            vid_hist[vid_label_gt][0]+=1
        else:
            dict_vid_stats['TF'] += 1
            vid_hist[vid_label_gt][1]+=1
    else:
        if vid_label_gt == vid_pred:
            dict_vid_stats['FT'] += 1
            vid_hist[vid_label_gt][2]+=1
        else:
            dict_vid_stats['FF'] += 1
            vid_hist[vid_label_gt][3]+=1

    seg_stats_counter = Counter(dict_segment_stats)
    vid_stats_counter = Counter(dict_vid_stats)

    return seg_stats_counter, vid_stats_counter, segment_hist, vid_hist


def write_eval_results(str_result_rootdir, class_name_list, mae_test, std_test, eval_results_prob, eval_results_mv, eval_results_prob_equal, eval_results_mv_equal, perf_prob, perf_majority, perf_prob_equal, perf_majority_equal):

    result_filepath = os.path.join(str_result_rootdir, "test_results.txt")

    print("*********************")
    print("For validation! TF and FT should be zero")
    print("Segment-based accuracy for the best model (equal weights) = {}".format(eval_results_prob_equal.get_seg_acc()*100))
    print("Segment-based F1 score for the best model (equal weights) = {}".format(eval_results_prob_equal.get_seg_f1score()*100))
    dict_segment_stats_prob_equal = perf_prob_equal.get_segment_transitions() 
    print("Segment transitions (equal weights) : TT / TF / FT / FF / Nof total segments = {} / {} / {} / {} / {}\n".format(dict_segment_stats_prob_equal['TT'], dict_segment_stats_prob_equal['TF'], dict_segment_stats_prob_equal['FT'], dict_segment_stats_prob_equal['FF'], sum(dict_segment_stats_prob_equal.values())))
    print("Video-based Accuracy for the best model (equal weights majority) = {}".format(eval_results_mv_equal.get_vid_acc()*100))
    print("Video-based F1 score for the best model (equal weights majority) = {}".format(eval_results_mv_equal.get_vid_f1score()*100))
    dict_vid_stats_majority_equal = perf_majority_equal.get_vid_transitions() 
    print("Video transitions (equal weights majority): TT / TF / FT / FF / Nof total videos = {} / {} / {} / {} / {}\n".format(dict_vid_stats_majority_equal['TT'], dict_vid_stats_majority_equal['TF'], dict_vid_stats_majority_equal['FT'], dict_vid_stats_majority_equal['FF'], sum(dict_vid_stats_majority_equal.values())))
    print("Video-based Accuracy for the best model (equal weights probs) = {}".format(eval_results_prob_equal.get_vid_acc()*100))
    print("Video-based F1 score for the best model (equal weights probs) = {}".format(eval_results_prob_equal.get_vid_f1score()*100))
    dict_vid_stats_prob_equal = perf_prob_equal.get_vid_transitions() 
    print("Video transitions (equal weights probs): TT / TF / FT / FF / Nof total videos = {} / {} / {} / {} / {}\n".format(dict_vid_stats_prob_equal['TT'], dict_vid_stats_prob_equal['TF'], dict_vid_stats_prob_equal['FT'], dict_vid_stats_prob_equal['FF'], sum(dict_vid_stats_prob_equal.values())))

    print("Model-based stream weighing performance evaluation")
    print("Test performances")
    print("Test MAE and STD = {:.2f} / {:.2f}".format(mae_test, std_test))

    print("*********************")
    print("Segment-based accuracy for the best model = {}".format(eval_results_prob.get_seg_acc()*100))
    print("Segment-based F1 score for the best model = {}".format(eval_results_prob.get_seg_f1score()*100))
    dict_segment_stats_prob = perf_prob.get_segment_transitions()
    print("Segment transitions: TT / TF / FT / FF / Nof total segments = {} / {} / {} / {} / {}\n".format(dict_segment_stats_prob['TT'], dict_segment_stats_prob['TF'], dict_segment_stats_prob['FT'], dict_segment_stats_prob['FF'], sum(dict_segment_stats_prob.values())))
    # print("Transition histograms for segments")
    # print("Class name = [TT TF FT FF]")
    # segment_hist_prob = perf_prob.get_segment_hist()
    # for class_idx, class_name in enumerate(class_name_list):
    #     print("{} = {}".format(class_name, segment_hist_prob[class_idx]))
    print("*********************")

    print("Video-based Accuracy for the best model (using majority voting) = {}".format(eval_results_mv.get_vid_acc()*100))
    print("Video-based F1 score for the best model (using majority voting) = {}".format(eval_results_mv.get_vid_f1score()*100))
    dict_vid_stats_majority = perf_majority.get_vid_transitions()
    print("Video transitions (using majority voting): TT / TF / FT / FF / Nof total videos = {} / {} / {} / {} / {}\n".format(dict_vid_stats_majority['TT'], dict_vid_stats_majority['TF'], dict_vid_stats_majority['FT'], dict_vid_stats_majority['FF'], sum(dict_vid_stats_majority.values())))
    # print("Transition histograms for videos (using majority voting)")
    # print("Class name = [TT TF FT FF]")
    # vid_hist_majority = perf_majority.get_vid_hist()
    # for class_idx, class_name in enumerate(class_name_list):
    #     print("{} = {}".format(class_name, vid_hist_majority[class_idx]))

    print("*********************")
    print("Video-based Accuracy for the best model (using sum of probabilities) = {}".format(eval_results_prob.get_vid_acc()*100))
    print("Video-based F1 score for the best model (using sum of probabilities) = {}".format(eval_results_prob.get_vid_f1score()*100))
    dict_vid_stats_prob = perf_prob.get_vid_transitions()
    print("Video transitions (using sum of probabilities): TT / TF / FT / FF / Nof total videos = {} / {} / {} / {} / {}\n".format(dict_vid_stats_prob['TT'], dict_vid_stats_prob['TF'], dict_vid_stats_prob['FT'], dict_vid_stats_prob['FF'], sum(dict_vid_stats_prob.values())))
    # print("Transition histograms for videos (using sum of probabilities)")
    # print("Class name = [TT TF FT FF]")
    # vid_hist_prob = perf_prob.get_vid_hist()
    # for class_idx, class_name in enumerate(class_name_list):
    #     print("{} = {}".format(class_name, vid_hist_prob[class_idx]))

    with open(result_filepath, 'w+') as result_file:

        result_file.write("*********************\n")
        result_file.write("{}\n".format(class_name_list))
        result_file.write("Video-based class accuracies (using majority voting) = \n")
        result_file.write("{}\n".format(eval_results_mv.get_vid_confmat().diagonal()))
        result_file.write("Video-based class accuracies (using sum of probabilities) = \n")
        result_file.write("{}\n".format(eval_results_prob.get_vid_confmat().diagonal()))
        result_file.write("{}\n".format(class_name_list))
        result_file.write("Video-based confusion matrix (using majority voting) = \n")
        result_file.write("{}\n".format(eval_results_mv.get_vid_confmat()))
        result_file.write("Video-based confusion matrix (using sum of probabilities) = \n")
        result_file.write("{}\n".format(eval_results_prob.get_vid_confmat()))
        result_file.write("{}\n".format(class_name_list))
        result_file.write("Segment-based class accuracies = \n")
        result_file.write("{}\n".format(eval_results_prob.get_seg_confmat().diagonal()))
        result_file.write("{}\n".format(class_name_list))
        result_file.write("Segment-based confusion matrix = \n")
        result_file.write("{}\n".format(eval_results_prob.get_seg_confmat()))

        result_file.write("Model-based stream weighing performance evaluation\n")
        result_file.write("Test performances\n")
        result_file.write("Test MAE and STD = {:.2f} / {:.2f}\n".format(mae_test, std_test))
        result_file.write("*********************\n")
        result_file.write("Segment-based accuracy for the best model = {}\n".format(eval_results_prob.get_seg_acc()*100))
        result_file.write("Segment-based F1 score for the best model = {}\n".format(eval_results_prob.get_seg_f1score()*100))
        result_file.write("Segment transitions: TT / TF / FT / FF / Nof total segments = {} / {} / {} / {} / {}\n".format(dict_segment_stats_prob['TT'], dict_segment_stats_prob['TF'], dict_segment_stats_prob['FT'], dict_segment_stats_prob['FF'], sum(dict_segment_stats_prob.values())))
        # result_file.write("Transition histograms for segments\n")
        # result_file.write("Class name = [TT TF FT FF]\n")
        # for class_idx, class_name in enumerate(class_name_list):
        #     result_file.write("{} = {}\n".format(class_name, segment_hist_prob[class_idx]))

        result_file.write("*********************\n")        
        result_file.write("Video-based Accuracy for the best model (using majority voting) = {}\n".format(eval_results_mv.get_vid_acc()*100))
        result_file.write("Video-based F1 score for the best model (using majority voting) = {}\n".format(eval_results_mv.get_vid_f1score()*100))
        result_file.write("Video transitions (using majority voting): TT / TF / FT / FF / Nof total videos = {} / {} / {} / {} / {}\n".format(dict_vid_stats_majority['TT'], dict_vid_stats_majority['TF'], dict_vid_stats_majority['FT'], dict_vid_stats_majority['FF'], sum(dict_vid_stats_majority.values())))
        # result_file.write("Transition histograms for videos (using majority voting)\n")
        # result_file.write("Class name = [TT TF FT FF]\n")
        # for class_idx, class_name in enumerate(class_name_list):
        #     result_file.write("{} = {}\n".format(class_name, vid_hist_majority[class_idx]))

        result_file.write("*********************\n")
        result_file.write("Video-based Accuracy for the best model (using sum of probabilities) = {}\n".format(eval_results_prob.get_vid_acc()*100))
        result_file.write("Video-based F1 score for the best model (using sum of probabilities) = {}\n".format(eval_results_prob.get_vid_f1score()*100))
        result_file.write("Video transitions (using sum of probabilities): TT / TF / FT / FF / Nof total videos = {} / {} / {} / {} / {}\n".format(dict_vid_stats_prob['TT'], dict_vid_stats_prob['TF'], dict_vid_stats_prob['FT'], dict_vid_stats_prob['FF'], sum(dict_vid_stats_prob.values())))
        # result_file.write("Transition histograms for videos (using sum of probabilities)\n")
        # result_file.write("Class name = [TT TF FT FF]\n")
        # for class_idx, class_name in enumerate(class_name_list):
        #     result_file.write("{} = {}\n".format(class_name, vid_hist_prob[class_idx]))

        result_file.write("*********************\n")
        result_file.write("For validation! TF and FT should be zero\n")
        result_file.write("Segment-based accuracy for the best model (equal weights) = {}\n".format(eval_results_prob_equal.get_seg_acc()*100))
        result_file.write("Segment-based F1 score for the best model (equal weights) = {}\n".format(eval_results_prob_equal.get_seg_f1score()*100))
        result_file.write("Segment transitions (equal weights) : TT / TF / FT / FF / Nof total segments = {} / {} / {} / {} / {}\n".format(dict_segment_stats_prob_equal['TT'], dict_segment_stats_prob_equal['TF'], dict_segment_stats_prob_equal['FT'], dict_segment_stats_prob_equal['FF'], sum(dict_segment_stats_prob_equal.values())))
        result_file.write("Video-based Accuracy for the best model (equal weights majority) = {}\n".format(eval_results_mv_equal.get_vid_acc()*100))
        result_file.write("Video-based F1 score for the best model (equal weights majority) = {}\n".format(eval_results_mv_equal.get_vid_f1score()*100))
        result_file.write("Video transitions (equal weights majority): TT / TF / FT / FF / Nof total videos = {} / {} / {} / {} / {}\n".format(dict_vid_stats_majority_equal['TT'], dict_vid_stats_majority_equal['TF'], dict_vid_stats_majority_equal['FT'], dict_vid_stats_majority_equal['FF'], sum(dict_vid_stats_majority_equal.values())))
        result_file.write("Video-based Accuracy for the best model (equal weights probs) = {}\n".format(eval_results_prob_equal.get_vid_acc()*100))
        result_file.write("Video-based F1 score for the best model (equal weights probs) = {}\n".format(eval_results_prob_equal.get_vid_f1score()*100))
        result_file.write("Video transitions (equal weights probs): TT / TF / FT / FF / Nof total videos = {} / {} / {} / {} / {}\n".format(dict_vid_stats_prob_equal['TT'], dict_vid_stats_prob_equal['TF'], dict_vid_stats_prob_equal['FT'], dict_vid_stats_prob_equal['FF'], sum(dict_vid_stats_prob_equal.values())))



if __name__ == '__main__':

    global args

    print("argument parser")    
    args = param_parser.parse_args()
    test_tsdf_fusion(args.config)