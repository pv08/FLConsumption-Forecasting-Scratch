import torch as T
from src.individual_training import IndividualTraining
from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default='dataset/full_dataset.csv')
    parser.add_argument("--data_path_test", type=list, default=['dataset/ElBorn_test.csv'])
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--targets", type=list, default=['rnti_count', 'rb_down', 'rb_up', 'down', 'up'])
    parser.add_argument("--num_lags", type=int, default=10)

    parser.add_argument("--filter_bs", type=any, default=None)
    parser.add_argument("--identifier", type=str, default='District')

    parser.add_argument("--nan_constant", type=int, default=0)
    parser.add_argument("--x_scaler", type=str, default='minmax')
    parser.add_argument("--y_scaler", type=str, default='minmax')
    parser.add_argument("--outlier_detection", type=any, default=None)

    parser.add_argument("--criterion", type=str, default='mse')
    parser.add_argument("--model_name", type=str, default='lstm', help='["mlp", "rnn" ,"lstm", "gru", "cnn", "da_encoder_decoder"]')
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--optimizer", type=str, default='adam')
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--early_stopping", type=bool, default=False)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--max_grad_norm", type=float, default=0.0)
    parser.add_argument("--reg1", type=float, default=0.0)  # l1 regularization
    parser.add_argument("--reg2", type=float, default=0.0)  # l2 regularization

    parser.add_argument("--plot_history", type=bool, default=True)  # plot loss history
    parser.add_argument("--cuda", type=bool, default=T.cuda.is_available())
    parser.add_argument("--seed", type=int, default=0)


    parser.add_argument("--assign_stats", type=any, default=None)  # whether to use statistics as exogenous data, ["mean", "median", "std", "variance", "kurtosis", "skew"]
    parser.add_argument("--use_time_features", type=bool, default=False)  # whether to use datetime features
    args = parser.parse_args()
    args.device = T.device('cuda' if T.cuda.is_available() else 'cpu')

    print(f"Script arguments {args}", end='\n')

    trainer = IndividualTraining(args=args)
    trained_model = trainer.fit()

if __name__ == "__main__":
    main()