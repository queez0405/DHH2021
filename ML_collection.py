import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.feature_selection import SelectFromModel

from sklearn.model_selection import KFold

from arguments import get_args
from tqdm import tqdm

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
    elif args.model == 'svr_rbf':
        ml_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
    elif args.model == 'svr_linear':
        ml_model = SVR(kernel='linear', C=100, gamma='auto')
    elif args.model == 'svr_poly':
        ml_model = SVR(kernel='poly', degree=2, C=100, gamma='auto')
    elif args.model == 'xgboost':
        ml_model = XGBRegressor(n_estimators=70, learning_rate=0.1, gamma=0, subsample=0.75, colsample_bytree=1, max_depth=7)
    elif args.model == 'adaboost':
        ml_model = AdaBoostRegressor(n_estimators=40, random_state=0)
    elif args.model == 'random_forest':
        ml_model = RandomForestRegressor(n_estimators=50, random_state=0)
    elif args.model == 'extra_tree':
        ml_model = ExtraTreesRegressor(n_estimators=50, random_state=1)

    return ml_model

sample_imp_regressor = ['adaboost', 'random_forest', 'extra_tree', 'decision_tree', 'bagging', 'gradient_boosting']

def main():
    clinical_csv = pd.read_csv('./dataset/Clinical_Variables.csv',index_col=0)
    genetic_csv = pd.read_csv('./dataset/Genetic_alterations.csv',index_col=0)
    surv_time_csv = pd.read_csv('./dataset/Survival_time_event.csv',index_col=0) 
    treat_csv = pd.read_csv('./dataset/Treatment.csv',index_col=0)

    args = get_args()
    
    if args.clinic_var_OH:
        clinical_OH_list = []
        for i in range(np.array(clinical_csv).shape[1]):
            clinical_OH_list.append(np.zeros([np.array(clinical_csv).shape[0], np.array(clinical_csv)[:,i].max()+1]))
        
        for i in range(np.array(clinical_csv).shape[1]):
            for j in range(len(clinical_csv)):
                clinical_OH_list[i][j, np.array(clinical_csv)[j,i]] = 1
            # coords = np.concatenate([np.arange(len(clinical_csv))[np.newaxis,:],np.array(clinical_csv)[:,i][np.newaxis,:]], axis=0).transpose()
            # clinical_OH_list[i][tuple([tuple(e) for e in coords])] = 1

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
        clinical_csv = clinical_csv.loc[is_treat,:]
        genetic_csv = genetic_csv.loc[is_treat,:]
        surv_time_csv = surv_time_csv.loc[is_treat,:]
        treat_csv = treat_csv.loc[is_treat,:]
    elif args.treat_type == 'not_treat_only':
        clinical_csv = clinical_csv.loc[is_not_treat,:]
        genetic_csv = genetic_csv.loc[is_not_treat,:]
        surv_time_csv = surv_time_csv.loc[is_not_treat,:]
        treat_csv = treat_csv.loc[is_not_treat,:]

    if args.treat_type == 'whole':
        x = np.concatenate((genetic_csv,clinical_csv,treat_csv), axis=1)
    else:
        x = np.concatenate((genetic_csv,clinical_csv), axis=1)
    y = np.array(surv_time_csv.loc[:,'time'])

    kfold = KFold(n_splits=10)
    ml_model = get_model(args)

    MSE = AverageMeter()
    MAE = AverageMeter()
    R2 = AverageMeter()
    importance = []

    for train_idx, test_idx in tqdm(kfold.split(x)):
        ml_model.fit(x[train_idx], y[train_idx])

        y_pred = ml_model.predict(x[test_idx])

        MSE.update(mean_squared_error(y_pred, y[test_idx]))
        MAE.update(mean_absolute_error(y_pred, y[test_idx]))
        R2.update(r2_score(y_pred, y[test_idx]))

        if args.model in sample_imp_regressor:
            importance.append(ml_model.feature_importances_[:300])

        # print(f'\nMSE:{mean_squared_error(y_pred, y[test_idx]):.03f}           MAE:{mean_absolute_error(y_pred, y[test_idx]):.03f}')
    if (args.model in sample_imp_regressor) and args.treat_type == 'not_treat_only':
        np.save('./not_treat_imp.npy', np.array(importance).mean(0))
    elif (args.model in sample_imp_regressor) and args.treat_type == 'treat_only':
        not_treat_imp = np.load('./not_treat_imp.npy')
        print((np.array(importance).mean(0) - not_treat_imp).argsort()[280:])
    print(f'MSE:{MSE.avg:.03f}           MAE:{MAE.avg:.03f}             R2:{R2.avg:.03f}')

if __name__ == '__main__':
    main()