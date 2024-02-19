import torch as T
from src.federated_training import FederatedTraining
from argparse import ArgumentParser
from src.utils.functions import plot_global_losses, plot_global_metrics, plot_local_train_rounds


def main():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default='dataset/pecanstreet/15min/')
    # parser.add_argument("--data_path_test", type=list, default=['dataset/ElBorn_test.csv'])
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--targets", type=list, default=['consumption']) # index 0
    parser.add_argument("--num_lags", type=int, default=10)

    parser.add_argument("--filter_bs", type=any, default=None)
    parser.add_argument("--identifier", type=str, default='cid')

    parser.add_argument("--nan_constant", type=int, default=0)
    parser.add_argument("--x_scaler", type=str, default='minmax')
    parser.add_argument("--y_scaler", type=str, default='minmax')
    parser.add_argument("--outlier_detection", type=any, default=None)

    parser.add_argument("--criterion", type=str, default='mse')
    parser.add_argument("--fl_rounds", type=int, default=10)
    parser.add_argument("--fraction", type=float, default=.25)
    parser.add_argument("--aggregation", type=str, default="fedavg")
    parser.add_argument("--model_name", type=str, default='cnn', help='["mlp", "rnn" ,"lstm", "gru", "cnn", "da_encoder_decoder"]')
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--optimizer", type=str, default='adam')
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--early_stopping", type=bool, default=False)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--max_grad_norm", type=float, default=0.0)
    parser.add_argument("--reg1", type=float, default=0.0)  # l1 regularization
    parser.add_argument("--reg2", type=float, default=0.0)  # l2 regularization


    parser.add_argument("--cuda", type=bool, default=T.cuda.is_available())
    parser.add_argument("--seed", type=int, default=0)


    parser.add_argument("--assign_stats", type=any, default=None)  # whether to use statistics as exogenous data, ["mean", "median", "std", "variance", "kurtosis", "skew"]
    parser.add_argument("--use_time_features", type=bool, default=False)  # whether to use datetime features
    args = parser.parse_args()
    args.device = T.device('cuda' if T.cuda.is_available() else 'cpu')

    print(f"Script arguments {args}", end='\n')

    local_train_params = {"epochs": args.epochs, "optimizer": args.optimizer, "lr": args.lr,
                          "criterion": args.criterion, "early_stopping": args.early_stopping,
                          "patience": args.patience, "device": args.device
                          }

    trainer = FederatedTraining(args=args)
    global_model, history = trainer.fit(local_train_params=local_train_params)
    trainer.evalutate_federated_model()
    history.save_in_json(model_name=args.model_name)
    print(history)
    plot_global_losses(history.global_test_losses)
    plot_global_metrics(history.global_test_metrics)
    plot_local_train_rounds(history.local_test_metrics)



if __name__ == "__main__":
    main()