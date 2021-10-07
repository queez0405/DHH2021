import os
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sklearn
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor, BaggingRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.model_selection import KFold

from arguments import get_args
from tqdm import tqdm

random_state = 1

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_model(args):
    if args.model == 'linear_reg':
        ml_model = LinearRegression()
        parameters = None
    elif args.model == 'svr_rbf':
        ml_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
        parameters = None
    elif args.model == 'svr_linear':
        ml_model = SVR(kernel='linear', C=100, gamma='auto')
        parameters = None
    elif args.model == 'svr_poly':
        ml_model = SVR(kernel='poly', degree=2, C=100, gamma='auto')
        parameters = None
    elif args.model == 'bagging':
        ml_model = BaggingRegressor(base_estimator=SVR(kernel = 'poly', degree=2, C=100, gamma='auto'), n_estimators=10, random_state=random_state)
        parameters = None
    elif args.model == 'xgboost':
        ml_model = XGBRegressor(booster='gbtree', n_estimators=250, learning_rate=0.08, gamma=0, subsample=0.75, max_depth=3)
        parameters = {
                        'n_estimators': [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250],
                        'max_depth': [None, 2, 3, 4, 5, 6, 7 , 8, 9, 10, 11, 12, 13, 14, 15, 16],
                        'learning_rate': [0.08, 0.09, 0.1, 0.11, 0.12],
                        'subsample': [0.25, 0.5, 0.75, 1.0],
                        'booster': ['gbtree', 'gblinear', 'dart'],
                        'gamma': [0, 0.001, 0.002, 0.01, 0.002, 0.1, 0.2]
                    }
    elif args.model == 'adaboost':
        ml_model = AdaBoostRegressor(learning_rate=1.2, loss='linear', n_estimators=100, random_state=random_state)
        parameters = {
                        'n_estimators': [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 200, 250],
                        'learning_rate': [0.08, 0.09, 0.1, 0.11, 0.12, 0.2, 0.5, 0.7, 1, 1.2, 1.5],
                        'loss': ['linear', 'square', 'exponential'],
                    }
    elif args.model == 'random_forest':
        ml_model = RandomForestRegressor(n_estimators=200, max_depth=15, max_features='auto', criterion='mse', bootstrap=True, random_state=random_state)
        parameters = {
                        'n_estimators': [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 200, 250],
                        'max_depth': [None, 2, 3, 4, 5, 6, 7 ,8, 9, 10, 11, 12, 13, 14, 15, 16],
                        'criterion': ['mse', 'mae'],
                        'max_features': ['auto', 'sqrt', 'log2'],
                        'bootstrap': [True, False],                        
                    }
    elif args.model == 'extra_tree':
        ml_model = ExtraTreesRegressor(n_estimators=100, max_depth=14, criterion='mse', max_features='auto', bootstrap='False', random_state=random_state)
        parameters = {
                        'n_estimators': [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 200, 250],
                        'max_depth': [None, 2, 3, 4, 5, 6, 7 ,8, 9, 10, 11, 12, 13, 14, 15, 16],
                        'criterion': ['mse', 'mae'],
                        'max_features': ['auto', 'sqrt', 'log2'],
                        'bootstrap': [True, False],                        
                    }
    elif args.model == 'decision_tree':
        ml_model = DecisionTreeRegressor(criterion = 'mae', max_depth=3, max_features='auto', random_state=random_state)
        parameters = {
                        'criterion': ['mse', 'friedman_mse', 'mae', 'poisson'],
                        'max_depth': [None, 2, 3, 4, 5, 6, 7 ,8, 9, 10, 11, 12, 13, 14, 15, 16],
                        'max_features': ['auto', 'sqrt', 'log2'],
                    }
    elif args.model == 'gradient_boosting':
        ml_model = GradientBoostingRegressor(n_estimators=250, learning_rate=0.11, max_depth=2, max_features='auto', criterion='friedman_mse', subsample=0.75, random_state=random_state)
        parameters = {
                        'n_estimators': [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 200, 250],
                        'learning_rate': [0.08, 0.09, 0.1, 0.11, 0.12, 0.2, 0.5, 0.7, 1, 1.2, 1.5],
                        'max_depth': [None, 2, 3, 4, 5, 6, 7 ,8, 9, 10, 11, 12, 13, 14, 15, 16],
                        'criterion': ['mse', 'friedman_mse', 'mae'],
                        'subsample': [0.25, 0.5, 0.75, 1.0],
                        'max_features': ['auto', 'sqrt', 'log2'],         
                    }
    elif args.model == 'lightgbm':
        ml_model = LGBMRegressor(n_estimators=250, learning_rate=0.12, max_depth=2, boosting_type='gbdt', subsample=1.0, random_state=random_state)
        parameters = {
                        'n_estimators': [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250],
                        'max_depth': [None, 2, 3, 4, 5, 6, 7 , 8, 9, 10, 11, 12, 13, 14, 15, 16],
                        'learning_rate': [0.08, 0.09, 0.1, 0.11, 0.12],
                        'subsample': [0.25, 0.5, 0.75, 1.0],
                        'boosting_type': ['gbdt', 'goss’', 'dart', 'rf'],
                    }
    
    if args.running_mode == 'search_param':
        return ml_model, parameters
    else:
        return ml_model

sample_imp_regressor = ['gradient_boosting', 'lightgbm', 'xgboost', 'random_forest']#'decision_tree', 'adaboost',

def get_whole_csv():
    clinical_csv = pd.read_csv('./dataset/Clinical_Variables.csv',index_col=0)
    genetic_csv = pd.read_csv('./dataset/Genetic_alterations.csv',index_col=0)
    surv_time_csv = pd.read_csv('./dataset/Survival_time_event.csv',index_col=0) 
    treat_csv = pd.read_csv('./dataset/Treatment.csv',index_col=0)

    return clinical_csv, genetic_csv, surv_time_csv, treat_csv

def get_treat_csv(is_treat, clinical_csv, genetic_csv, surv_time_csv, treat_csv):
    treated_clinical_csv = clinical_csv.loc[is_treat,:]
    treated_genetic_csv = genetic_csv.loc[is_treat,:]
    treated_surv_time_csv = surv_time_csv.loc[is_treat,:]
    treated_treat_csv = treat_csv.loc[is_treat,:]

    return treated_clinical_csv, treated_genetic_csv, treated_surv_time_csv, treated_treat_csv

def get_not_treat_csv(is_not_treat, clinical_csv, genetic_csv, surv_time_csv, treat_csv):
    not_treated_clinical_csv = clinical_csv.loc[is_not_treat,:]
    not_treated_genetic_csv = genetic_csv.loc[is_not_treat,:]
    not_treated_surv_time_csv = surv_time_csv.loc[is_not_treat,:]
    not_treated_treat_csv = treat_csv.loc[is_not_treat,:]

    return not_treated_clinical_csv, not_treated_genetic_csv, not_treated_surv_time_csv, not_treated_treat_csv

def main():
    clinical_csv, genetic_csv, surv_time_csv, treat_csv = get_whole_csv()

    args = get_args()
    
    if args.clinic_var_OH:
        clinical_OH_list = []
        for i in range(np.array(clinical_csv).shape[1]):
            clinical_OH_list.append(np.zeros([np.array(clinical_csv).shape[0], np.array(clinical_csv)[:,i].max()+1]))
        
        for i in range(np.array(clinical_csv).shape[1]):
            for j in range(len(clinical_csv)):
                clinical_OH_list[i][j, np.array(clinical_csv)[j,i]] = 1

        clinical_csv = clinical_OH_list[0]
        for i in range(1,len(clinical_OH_list)):
            clinical_csv = np.concatenate((clinical_csv,clinical_OH_list[i]), axis=1)
    
    is_survive = surv_time_csv['event'] == 0
    is_dead = surv_time_csv['event'] == 1

    is_not_treat = treat_csv['Treatment'] == 0
    is_treat = treat_csv['Treatment'] == 1

    if args.event_preprocess == 'same':
       surv_time_csv.loc[is_survive, 'time'] += (surv_time_csv.loc[is_dead, 'time'].mean() - surv_time_csv.loc[is_survive, 'time'].mean())
    elif args.event_preprocess == 'diff':
        surv_time_diff_treat = surv_time_csv[is_dead & is_treat]['time'].mean() - surv_time_csv[is_survive & is_treat]['time'].mean()
        surv_time_csv.loc[is_survive & is_treat,'time'] += surv_time_diff_treat
        surv_time_diff_not_treat = (surv_time_csv[is_dead & is_not_treat]['time'].mean() - surv_time_csv[is_survive & is_not_treat]['time'].mean())
        surv_time_csv.loc[is_survive & is_not_treat, 'time']  += surv_time_diff_not_treat

    if args.treat_type == 'treat_only':
        clinical_csv, genetic_csv, surv_time_csv, treat_csv = get_treat_csv(is_treat, clinical_csv, genetic_csv, surv_time_csv, treat_csv)
    elif args.treat_type == 'not_treat_only':
        clinical_csv, genetic_csv, surv_time_csv, treat_csv = get_not_treat_csv(is_not_treat, clinical_csv, genetic_csv, surv_time_csv, treat_csv)
        
    if args.treat_type == 'whole':
        x = np.concatenate((genetic_csv,clinical_csv,treat_csv), axis=1)
    else:
        x = np.concatenate((genetic_csv,clinical_csv), axis=1)
    
    y = np.array(surv_time_csv.loc[:,'time'])

    RMS = AverageMeter()
    MAE = AverageMeter()
    R2 = AverageMeter()

    if args.running_mode == 'show_perform':
        kfold = KFold(n_splits=10, random_state=random_state, shuffle=True)
        ml_model = get_model(args)

        for train_idx, test_idx in tqdm(kfold.split(x)):
            ml_model.fit(x[train_idx], y[train_idx])

            y_pred = ml_model.predict(x[test_idx])

            RMS.update(mean_squared_error(y_pred, y[test_idx])**0.5)
            MAE.update(mean_absolute_error(y_pred, y[test_idx]))
            R2.update(r2_score(y_pred, y[test_idx]))

        print(f'RMS:{RMS.avg:.03f}           MAE:{MAE.avg:.03f}             R2:{R2.avg:.03f}')
        print(RMS.avg)
        print(MAE.avg)
        print(R2.avg)
    elif args.running_mode == 'search_param':
        kfold = KFold(n_splits=10, random_state=random_state, shuffle=True)
        with open('./result.txt', mode = 'w') as f:
            f.write(str(datetime.datetime.now())+'\n')
        for model in sample_imp_regressor:
            args.model = model
            ml_model, parameters = get_model(args)
            if args.search_type == 'grid':
                grid_search = GridSearchCV(estimator=ml_model,
                                param_grid=parameters,
                                cv=kfold,
                                n_jobs=-1,
                                scoring='neg_mean_squared_error',
                                verbose=2
                                )
                grid_search.fit(x,y)
                with open('./result.txt', mode = 'a') as f:
                    f.write(str(grid_search.best_estimator_)+'\n')
                    f.write(str(grid_search.best_params_)+'\n')
                    f.write(str(grid_search.best_score_)+'\n')
            elif args.search_type == 'random':
                random_search = RandomizedSearchCV(estimator=ml_model,
                                param_distributions=parameters,
                                cv=kfold,
                                n_jobs=-1,
                                scoring='neg_mean_squared_error',
                                verbose=2
                                )
                best_score = -1000.
                while True:
                    random_search.fit(x,y)
                    print(random_search.best_params_)
                    print(random_search.best_score_)
                    if random_search.best_score_ > best_score:
                        best_score = random_search.best_score_
                        with open('./random_result.txt', mode = 'a') as f:
                            f.write(str(random_search.best_params_)+'\n')
                            f.write(str(best_score)+'\n')
                print(random_search.best_estimator_)

    elif args.running_mode == 'select_gene':
        not_treated_clinical_csv, not_treated_genetic_csv, not_treated_surv_time_csv, not_treated_treat_csv = get_not_treat_csv(is_not_treat, clinical_csv, genetic_csv, surv_time_csv, treat_csv)
        not_treated_x = np.concatenate((not_treated_genetic_csv,not_treated_clinical_csv), axis=1)
        not_treated_y = np.array(not_treated_surv_time_csv.loc[:,'time'])
        not_treated_importance = []

        treated_clinical_csv, treated_genetic_csv, treated_surv_time_csv, treated_treat_csv = get_treat_csv(is_treat, clinical_csv, genetic_csv, surv_time_csv, treat_csv)
        treated_x = np.concatenate((treated_genetic_csv,treated_clinical_csv), axis=1)
        treated_y = np.array(treated_surv_time_csv.loc[:,'time'])
        
        treated_importance = []

        point_dic = {i:[] for i in range(genetic_csv.shape[1])}
        weighted_value = []

        for model in sample_imp_regressor:

            print(model)
            args.model = model
            ml_model = get_model(args)

            kfold = KFold(n_splits=10, random_state=random_state, shuffle=True)

            RMS.reset()
            MAE.reset()
            R2.reset()

            for train_idx, test_idx in tqdm(kfold.split(x)):
                ml_model.fit(x[train_idx], y[train_idx])

                y_pred = ml_model.predict(x[test_idx])

                RMS.update(mean_squared_error(y_pred, y[test_idx])**0.5)
                MAE.update(mean_absolute_error(y_pred, y[test_idx]))
                R2.update(r2_score(y_pred, y[test_idx]))
            print(f'RMS:{RMS.avg:.03f}           MAE:{MAE.avg:.03f}             R2:{R2.avg:.03f}')
            weighted_value.append(RMS.avg)

            RMS.reset()
            MAE.reset()
            R2.reset()

            for train_idx, test_idx in tqdm(kfold.split(not_treated_x)):
                ml_model.fit(not_treated_x[train_idx], not_treated_y[train_idx])

                y_pred = ml_model.predict(not_treated_x[test_idx])

                RMS.update(mean_squared_error(y_pred, not_treated_y[test_idx])**0.5)
                MAE.update(mean_absolute_error(y_pred, not_treated_y[test_idx]))
                R2.update(r2_score(y_pred, not_treated_y[test_idx]))

                not_treated_importance.append(ml_model.feature_importances_[:genetic_csv.shape[1]])
            # print(f'RMS:{RMS.avg:.03f}           MAE:{MAE.avg:.03f}             R2:{R2.avg:.03f}')

            RMS.reset()
            MAE.reset()
            R2.reset()

            for train_idx, test_idx in tqdm(kfold.split(treated_x)):
                ml_model.fit(treated_x[train_idx], treated_y[train_idx])

                y_pred = ml_model.predict(treated_x[test_idx])

                RMS.update(mean_squared_error(y_pred, treated_y[test_idx])**0.5)
                MAE.update(mean_absolute_error(y_pred, treated_y[test_idx]))
                R2.update(r2_score(y_pred, not_treated_y[test_idx]))

                treated_importance.append(ml_model.feature_importances_[:genetic_csv.shape[1]])
            # print(f'RMS:{RMS.avg:.03f}           MAE:{MAE.avg:.03f}             R2:{R2.avg:.03f}')

            importance_ranking = (np.array(treated_importance).mean(0) - np.array(not_treated_importance).mean(0)).argsort()
            for rank, gene_num in enumerate(importance_ranking):
                point_dic[gene_num].append(rank+1)

        ##### add transformer result
        trans_ranking = np.flip(np.array([295,179,81,35,289,2,18,106,261,147,41,161,242,32,290,199,196,162,108,188,95,213,5,17,218,79,243,154,167,206,44,225,192,215,214,165,135,30,72,185,241,42,117,84,132,277,257,25,103,65,291,24,61,151,100,267,92,168,69,238,239,198,78,184,10,193,7,110,264,297,223,89,46,37,229,27,227,43,55,149,115,235,262,153,296,240,31,124,29,15,232,220,285,204,205,200,98,268,148,67,283,128,266,3,66,271,172,203,12,104,251,248,278,157,253,34,109,228,13,233,63,94,221,96,141,234,282,237,20,113,163,51,259,217,116,293,300,52,208,73,284,45,127,255,216,4,265,250,19,142,189,8,236,279,130,83,120,114,107,182,122,299,286,152,134,294,101,36,195,176,133,129,150,270,181,76,99,87,164,180,207,21,269,158,22,276,47,136,209,86,11,57,91,211,93,54,70,77,49,102,6,171,190,210,252,105,131,58,260,186,14,143,38,222,75,62,125,126,219,123,166,224,53,178,155,111,191,26,212,230,48,231,170,275,246,40,39,71,16,256,156,144,201,254,169,183,80,245,272,64,194,292,33,138,173,280,287,288,249,60,85,175,112,281,139,258,146,263,187,140,226,59,90,244,273,298,74,174,97,160,68,56,202,9,82,137,145,50,23,274,247,177,119,197,159,1,88,118,28,121])-1).tolist()
        trans_rms = 7.227466821670532
        for rank, gene_num in enumerate(trans_ranking):
            point_dic[gene_num].append(rank+1)

        weighted_value.append(trans_rms)

        weighted_value = [1/i for i in weighted_value]
        weighted_value = [float(i)/sum(weighted_value) for i in weighted_value]
        for i, rankings in enumerate(point_dic.values()):
            point_dic[i] = np.average(point_dic[i], weights=weighted_value)

        result = sorted(point_dic.items(), key=lambda x:x[1])

        print(np.flip(np.array([i for i,_ in result])+1).tolist())
        
    else:
        return

if __name__ == '__main__':
    main()