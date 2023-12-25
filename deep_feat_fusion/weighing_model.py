import os
import pickle
# Regression models
from sklearn.linear_model import LinearRegression, Ridge, MultiTaskLassoCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor, GradientBoostingRegressor, StackingRegressor, VotingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from ngboost import NGBRegressor
import xgboost as xgb


class WeighingModel:
    """Create a Weighing Model Object 

    Args:
        str_config: The path of the configuration file
    """

    def __init__(self):
        super(WeighingModel, self).__init__()
        # Regression model for weighing
        self.weighing_model = None
        self.scaler = None

        # Load previously trained weighing model
        def load_weighing_model(self, model_filepath, scaler_filepath):

            if os.path.isfile(model_filepath) or os.path.isfile(scaler_filepath): # read from file
                # print("Weighing model has been trained before. Thus, reading from file")
                print("Weighing model filepath = ", model_filepath)
                with open(model_filepath, 'rb') as model_file:
                    self.weighing_model = pickle.load(model_file)

                with open(scaler_filepath, 'rb') as scaler_file:
                    self.scaler = pickle.load(scaler_file)
            else:
                raise ValueError("One of the files {}/{} does not exist".format(model_filepath, scaler_filepath))


        def predict(self):

            if self.weighing_model == None:
                raise ValueError("Weighing model should be loaded at first for weight prediction!")

            pred_weights = self.swm_model.predict(feats)

            return pred_weights 


    # def train_swm(self, feats, targets):

        #   self.weighing_model = get_regression_model(str_model_type)
    #     swm_model_loaded = self.load_weighing_models(self.swm_model_filepath)
    #     dict_swm_feats_train, dict_swm_targets_train = self.load_weighing_feats(self.swm_trainfeat_filepath) 
    #     scaler_fusion = self.load_weighing_scaler(self.swmscaler_filepath)

    #     if dict_swm_feats_train is None:
    #         # Read training features for SWM and extract scaler
    #         dict_arn_train_feats, \
    #         dict_arn_train_labels = self.read_vid_features("trainval")

    #         dict_swm_feats_train, \
    #         dict_swm_targets_train = self.compute_swm_targets(dict_arn_train_feats, dict_arn_train_labels, "normalized")                
    #         # Scale train features
    #         dict_swm_feats_train, scaler_fusion = self.scale_features(dict_swm_feats_train, None, self.swm_feat_scaler_type)                                
    #     else:
    #         if scaler_fusion is None:
    #             dict_swm_feats_train, scaler_fusion = self.scale_features(dict_swm_feats_train, None, self.swm_feat_scaler_type)                                
    #         else:
    #             dict_swm_feats_train, _ = self.scale_features(dict_swm_feats_train, scaler_fusion, None)

    #     # Save features which are not scaled (this data will be used with other scalers after)  
    #     print("saving feats to file = ", self.swm_trainfeat_filepath)
    #     with open(self.swm_trainfeat_filepath, 'wb') as swm_trainfeat_file:
    #         pickle.dump(dict_swm_feats_train, swm_trainfeat_file, pickle.HIGHEST_PROTOCOL)
    #         pickle.dump(dict_swm_targets_train, swm_trainfeat_file, pickle.HIGHEST_PROTOCOL)
    #     # Save scaler
    #     print("saving scaler to file = ", self.swmscaler_filepath)
    #     with open(self.swmscaler_filepath, 'wb') as swmscaler_file:
    #         pickle.dump(scaler_fusion, swmscaler_file, pickle.HIGHEST_PROTOCOL)
    
    #     # If model is not previously trained 
    #     if swm_model_loaded is None:
    #         self.swm_model = self.train_weighing_model(dict_swm_feats_train, dict_swm_targets_train, "swm")
    #         # save attention weights into stream_attention_weights
    #         with open(self.swm_model_filepath, 'wb') as swm_model_file:
    #             pickle.dump(self.swm_model, swm_model_file, pickle.HIGHEST_PROTOCOL)
    #     else:
    #         self.swm_model = swm_model_loaded

    #     mse_train, mae_train, std_train = self.plot_regression_diff("train", dict_swm_feats_train, dict_swm_targets_train)
    #     self.write_train_results(mae_train, std_train)


    # def train_dwm(self):

    #     dwm_model_loaded = self.load_weighing_models(self.dwm_model_filepath)
    #     dict_dwm_feats_train, dict_dwm_targets_train = self.load_weighing_feats(self.dwm_trainfeat_filepath) 
    #     scaler_fusion = self.load_weighing_scaler(self.dwmscaler_filepath)

    #     # Scaler should be defined to scale test features
    #     if scaler_fusion is None or dict_dwm_feats_train is None:
    #         if dict_dwm_feats_train is None:
    #             # Read training features for DWM and extract scaler
    #             print("reading training samples...")
    #             dict_arn_train_feats, \
    #             dict_arn_train_labels = self.read_vid_features("trainval")

    #             print("computing DWM weights")
    #             dict_dwm_feats_train, \
    #             dict_dwm_targets_train = self.compute_dwm_targets(dict_arn_train_feats, dict_arn_train_labels, "train")                
    #             # Scale train features
    #             dict_dwm_feats_train, scaler_fusion = self.scale_features(dict_dwm_feats_train, None, self.dwm_feat_scaler_type)                                
    #         else:
    #             if scaler_fusion is None:
    #                 dict_dwm_feats_train, scaler_fusion = self.scale_features(dict_dwm_feats_train, None, self.dwm_feat_scaler_type)                                
    #             else:
    #                 dict_dwm_feats_train, _ = self.scale_features(dict_dwm_feats_train, scaler_fusion, None)

    #         # Save features which are not scaled (this data will be used with other scalers after)  
    #         print("saving feats to file = ", self.dwm_trainfeat_filepath)
    #         with open(self.dwm_trainfeat_filepath, 'wb') as dwm_trainfeat_file:
    #             pickle.dump(dict_dwm_feats_train, dwm_trainfeat_file, pickle.HIGHEST_PROTOCOL)
    #             pickle.dump(dict_dwm_targets_train, dwm_trainfeat_file, pickle.HIGHEST_PROTOCOL)
    #         # Save scaler
    #         print("saving scaler to file = ", self.dwmscaler_filepath)
    #         with open(self.dwmscaler_filepath, 'wb') as dwmscaler_file:
    #             pickle.dump(scaler_fusion, dwmscaler_file, pickle.HIGHEST_PROTOCOL)
    
    #     # If model is not previously trained 
    #     if dwm_model_loaded is None:
    #         self.dwm_model = self.train_weighing_model(dict_dwm_feats_train, dict_dwm_targets_train, "dwm")
    #         # save attention weights into stream_attention_weights
    #         with open(self.dwm_model_filepath, 'wb') as dwm_model_file:
    #             pickle.dump(self.dwm_model, dwm_model_file, pickle.HIGHEST_PROTOCOL)
    #     else:
    #         self.dwm_model = dwm_model_loaded


    def get_regression_model(self, str_model_type):

        if str_model_type == "linear":
            print("Regression type is linear")
            weighing_model = LinearRegression(n_jobs=nof_jobs, positive=True)
        elif str_model_type == "kneighbor":
            print("Regression type is k-nearest")
            weighing_model = KNeighborsRegressor(n_jobs=nof_jobs, n_neighbors=10)
        elif str_model_type == "decisiontree":
            print("Regression type is decision tree")
            weighing_model = DecisionTreeRegressor()
        elif str_model_type == "randomforest":
            print("Regression type is random forest")
            weighing_model = RandomForestRegressor(n_jobs=nof_jobs)
        elif (str_model_type == 'ridge'):
            weighing_model_tmp = Ridge()
            cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
            grid = dict()
            grid['alpha'] = np.arange(0.1, 25, 5)
            weighing_model = GridSearchCV(weighing_model_tmp, grid, scoring='neg_mean_absolute_error', cv=cv, n_jobs=10)         
        elif (str_model_type == 'lasso'):
            cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
            weighing_model = MultiTaskLassoCV(alphas=np.arange(0.01, 1, 0.01), tol=0.01, cv=cv, n_jobs=4)
        elif (str_model_type == 'multiout_svrlinear') or \
                (str_model_type == 'multiout_svrrbf') or \
                (str_model_type == 'multiout_svrpoly') or \
                (str_model_type == 'multiout_ngboost') or \
                (str_model_type == 'multiout_xgboost'):

            print("Regression type is multiout")
            if (str_model_type == 'multiout_svrlinear'):
                print("Regression type is multiout linear svr")
                model = LinearSVR()
            elif (str_model_type == 'multiout_svrrbf'):
                print("Regression type is multiout rbf svr")
                model = SVR(kernel='rbf')
            elif (str_model_type == 'multiout_svrpoly'):
                print("Regression type is multiout polynomial svr")
                model = SVR(kernel='poly')
            elif (str_model_type == 'multiout_ngboost'):
                print("Regression type is multiout ngboost")
                model = NGBRegressor()
            elif (str_model_type == 'multiout_xgboost'):
                print("Regression type is multiout xgboost")
                print("Performing hyperparameter search!")
                model_xgboost = xgb.XGBRegressor(eval_metric='mae', objective='reg:squarederror', random_state=0)
                distributions = dict(reg_lambda = [3, 5],
                                    scale_pos_weight = [0.1, 0.5, 0.8], 
                                    eta=[0.01, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.2, 0.25, 0.3, 0.35, 0.40, 0.45, 0.5],
                                    max_depth=[4, 5, 6, 7, 8],
                                    objective=['reg:squarederror', 'reg:squaredlogerror', 'reg:logistic', 'reg:pseudohubererror', 'reg:gamma'],
                                    reg_alpha=[1, 3, 5, 10])

                model = RandomizedSearchCV(model_xgboost, distributions, n_jobs=-1)

            weighing_model = MultiOutputRegressor(model)

        elif (str_model_type == 'chain_svrlinear') or \
                (str_model_type == 'chain_svrrbf') or \
                (str_model_type == 'chain_svrpoly'):

            print("Regression type is chain")
            if (str_model_type == 'chain_svrlinear'):
                model = LinearSVR()
            elif (str_model_type == 'chain_svrrbf'):
                model = SVR(kernel='rbf')
            elif (str_model_type == 'chain_svrpoly'):
                model = SVR(kernel='poly')

            weighing_model = RegressorChain(model)
            
        else:
            raise ValueError("Unknown regression type for stream attention training!")

        return weighing_model
