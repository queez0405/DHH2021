import argparse

def get_args():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--running_mode',type=str, default='show_perform', choices=['show_perform', 'search_param', 'select_gene'])
    parser.add_argument('--search_type',type=str, default='gird', choices=['grid','random'])

    parser.add_argument('--model',type=str, default='svr_poly',
                                            choices=['linear_reg', 'random_forest', 'extra_tree', 'decision_tree', 'bagging', 'gradient_boosting', 'svr_rbf', 'svr_linear', 'svr_poly', 'adaboost', 'xgboost', 'lightgbm'])

    # dataset option
    parser.add_argument('--clinic_var_OH',type=bool, default=False)
    parser.add_argument('--treat_type',type=str, default='whole', choices=['whole', 'treat_only', 'not_treat_only'])
    parser.add_argument('--event_preprocess',type=str, default='same', choices=['none', 'same', 'diff'])
    parser.add_argument('--surv_time_norm',type=bool, default=False)

    # option for grid search

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')

    opt = parser.parse_args()
    opt.save_folder = './save/'

    return opt