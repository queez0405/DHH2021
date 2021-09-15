import argparse

def get_args():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of training epochs')
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--dropout',type=float, default=0.1)

    parser.add_argument('--model',type=str, default='svr_poly', choices=['linear_reg', 'random_forest', 'extra_tree', 'svr_rbf', 'svr_linear', 'svr_poly', 'adaboost', 'xgboost'])

    # dataset option
    parser.add_argument('--clinic_var_OH',type=bool, default=False)
    parser.add_argument('--treat_type',type=str, default='whole', choices=['whole', 'treat_only', 'not_treat_only'])
    parser.add_argument('--event_preprocess',type=str, default='same', choices=['none', 'same', 'diff'])
    parser.add_argument('--surv_time_norm',type=bool, default=False)

    # optimization
    parser.add_argument('--optimizer',type=str, default='sgd',choices=['sgd','adam'])
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='350,400,450',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')


    opt = parser.parse_args()
    opt.save_folder = './save/'

    return opt