import copy
import torch as T
import wandb
import pandas as pd
from logging import INFO
from src.utils.logger import log
from src.utils.functions import plot_test_prediction
from collections import OrderedDict
from src.base.trainers import Trainers
from src.fl.federated_learning import FederatedLearning
from src.dataset.processing import Processsing
from src.data import TimeSeriesLoader
from src.fl.client_proxy import SimpleClientProxy
from src.fl.server.server import Server

class FederatedTraining(Trainers):
    def __init__(self, args):
        super(FederatedTraining, self).__init__(args=args)
        self.wandb_logger = wandb.init(project='FL-ConsumptionForecasting-Scratch', config=self.args,
                                       tags=['FL', 'consumption', self.args.model_name], group='Federated', name=f"{self.args.model_name}_Federated")
        self.seed_all(args.seed)
        processing = Processsing(args=self.args, data_path=self.args.data_path)

        X_train, X_val, X_test, y_train, y_val, y_test, exogenous_data_train, exogenous_data_val, x_scaler, y_scaler = processing.make_preprocessing(per_area=True)

        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test, self.client_X_train, self.client_X_val, self.client_X_test, self.client_y_train, self.client_y_val, self.client_y_test, self.exogenous_data_train, self.exogenous_data_val = (
            processing.make_postprocessing(X_train, X_val, X_test, y_train, y_val, y_test, exogenous_data_train,
                                           exogenous_data_val, x_scaler, y_scaler))

        if len(self.client_X_train.keys()) == len(self.client_y_val.keys()):
            for client in self.client_X_train:
                print(f"\nClient: {client}")
                print(f"\tX_train shape: {self.client_X_train[client].shape}, y_train shape: {self.client_y_train[client].shape}")
                print(f"\tX_val shape: {self.client_X_val[client].shape}, y_val shape: {self.client_y_val[client].shape}")
                print(f"\tX_test shape: {self.client_X_test[client].shape}, y_val shape: {self.client_y_test[client].shape}")
        else:
            raise InterruptedError(f"[!] - Length of clients not equivalent")


        input_dim, exogenous_dim = processing.get_input_dims(self.X_train, self.exogenous_data_train)

        print(input_dim, exogenous_dim)
        self.model = self.get_model(model=args.model_name,
                          input_dim=input_dim,
                          out_dim=self.y_train.shape[1],
                          lags=args.num_lags,
                          exogenous_dim=exogenous_dim,
                          seed=args.seed)



    def fit(self, idxs=[0],
            log_per=1,
            client_creation_fn = None, # client specification
            local_train_params=None, # local params
            aggregation_params=None, # aggregation params
            use_carbontracker=False):
        if client_creation_fn is None:
            client_creation_fn = FederatedLearning.create_regression_client

        if local_train_params is None:
            local_train_params = {"epochs": self.args.epochs, "optimizer": self.args.optimizer,
                                  "lr": self.args.lr, "criterion": self.args.criterion, "early_stopping": self.args.early_stopping,
                                  "patince": self.args.patience, "device": self.args.device}

        train_loaders, val_loaders = [], []

        for client in self.client_X_train:
            if client == "all":
                continue
            if self.exogenous_data_train is not None:
                tmp_exogenous_data_train = self.exogenous_data_train[client]
                tmp_exogenous_data_val = self.exogenous_data_val[client]
            else:
                tmp_exogenous_data_train = None
                tmp_exogenous_data_val = None
            num_features = len(self.client_X_train[client][0][0])

            train_loaders.append(
                TimeSeriesLoader(X=self.client_X_train[client],
                                 y=self.client_y_train[client],
                                 num_lags=self.args.num_lags,
                                 num_features=num_features, exogenous_data=tmp_exogenous_data_train,
                                 indices=idxs, batch_size=self.args.batch_size, shuffle=False).get_dataloader()
            )
            val_loaders.append(
                TimeSeriesLoader(X=self.client_X_val[client],
                                 y=self.client_y_val[client],
                                 num_lags=self.args.num_lags,
                                 num_features=num_features, exogenous_data=tmp_exogenous_data_val,
                                 indices=idxs, batch_size=self.args.batch_size, shuffle=False).get_dataloader()
            )

        self.cids = [k for k in self.client_X_train.keys() if k != 'all']

        clients = [
            client_creation_fn(cid=cid, model=self.model,
                               train_loader=train_loader, test_loader=val_loader,
                               local_params=local_train_params)
            for cid, train_loader, val_loader in zip(self.cids, train_loaders, val_loaders)
        ]

        client_proxies = [SimpleClientProxy(cid, client) for cid, client in zip(self.cids, clients)]
        server = Server(client_proxies=client_proxies,
                        aggregation=self.args.aggregation,
                        aggregation_params=aggregation_params,
                        local_params_fn=None, server_model=self.model, server_config=self.args, logger=self.wandb_logger)

        model_params, history = server.fit(self.args.fl_rounds, self.args.fraction, use_carbotracker=use_carbontracker)
        params_dict = zip(self.model.state_dict().keys(), model_params)
        state_dict = OrderedDict({k: T.Tensor(v) for k, v in params_dict})
        self.model = copy.deepcopy(self.model)
        self.model.load_state_dict(state_dict, strict=True)
        return self.model, history


    def evalutate_federated_model(self):
        evaluate_metrics = []
        for client in self.cids:
            num_features = len(self.client_X_test[client][0][0])
            test_loader = TimeSeriesLoader(X=self.client_X_test[client],
                                 y=self.client_y_test[client],
                                 num_lags=self.args.num_lags,
                                 num_features=num_features, exogenous_data=None,
                                 indices=[0], batch_size=1, shuffle=False).get_dataloader()
            test_mse, test_rmse, test_mae, test_r2, test_nrmse, test_pinball, y_pred_test = Trainers(args=self.args).test(
                self.model, test_loader, None, device=self.args.device
            )
            evaluate_metrics.append({'cid': client, 'mse': test_mse, 'mae': test_mae, 'r2': test_r2, 'pinball': test_pinball})
            log(INFO, f"Client: {client} | MSE: {test_mse} | MAE: {test_mae} | pinball loss: {test_pinball}")
            plot_test_prediction(y_true=self.client_y_test[client], y_pred=y_pred_test, cid=client)
        evaluate_metrics_df = pd.DataFrame(evaluate_metrics)
        evaluate_metrics_df.to_csv('etc/results/eval_metrics.csv')
