from typing import Dict

import torch as T
import flwr as fl
from flwr.common import Scalar, NDArrays

from src.base.trainers import Trainers
from src.dataset.processing import Processsing
from src.data import TimeSeriesLoader
from collections import OrderedDict



class CifarClient(fl.client.NumPyClient):
    def __init__(self, args, inputdim, outdim, exogenous_dim, trainset, valset, testset, device: str):
        self.args = args
        self.inputdim = inputdim
        self.outdim = outdim
        self.exogenous_dim = exogenous_dim

        self.device = device
        self.trainset = trainset
        self.valset = valset
        self.testset = testset
        self.trainer = Trainers(args=args)

    def set_parameters(self, parameters):
        """Loads a efficientnet model and replaces it parameters with the ones given."""
        model = self.trainer.get_model(model=self.args.model_name,
                          input_dim=self.inputdim,
                          out_dim=self.outdim,
                          lags=self.args.num_lags,
                          exogenous_dim=self.exogenous_dim,
                          seed=self.args.seed)
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: T.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        return model

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        # Update local model parameters
        model = self.set_parameters(parameters)


        model = self.trainer.train(model=model, train_loader=self.trainset, test_loader=self.valset,
                      epochs=self.args.epochs,
                      optimizer=self.args.optimizer, lr=self.args.lr,
                      criterion=self.args.criterion,
                      early_stopping=self.args.early_stopping,
                      patience=self.args.patience,
                      device=self.args.device, cid=self.args.filter_bs)
        criterion = self.trainer.get_criterion()
        test_loss, test_mse, test_rmse, test_mae, test_r2, test_nrmse = self.trainer.train(model, self.valset,
                                                                                 criterion, self.args.device)

        parameters_prime = [val.cpu().numpy() for _, val in model.state_dict().items()]

        return parameters_prime, len(self.trainset), {'loss': float(test_loss), 'mse': float(test_mse)}

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""
        # Update local model parameters
        model = self.set_parameters(parameters)

        # Get config values
        criterion = self.trainer.get_criterion()
        test_loss, test_mse, test_rmse, test_mae, test_r2, test_nrmse = self.trainer.train(model, self.valset,
                                                                                 criterion, self.args.device)
        return float(test_loss), len(self.testset), {'loss': float(test_loss), 'mse': float(test_mse)}



class IndividualTraining(Trainers):
    def __init__(self, args):
        super(IndividualTraining, self).__init__(args=args)
        if args.outlier_detection is not None:
            outlier_columns = ['rb_down', 'rb_up', 'down', 'up']
            outlier_kwargs = {"ElBorn": (10, 10), "LesCorts": (10, 90), "PobleSec": (5, 95)}
            args.outlier_columns = outlier_columns
            args.outlier_kwargs = outlier_kwargs

        self.seed_all(args.seed)

        processing = Processsing(args=self.args, data_path=self.args.data_path)

        X_train, X_val, X_test, y_train, y_val, y_test, exogenous_data_train, exogenous_data_val, x_scaler, y_scaler = processing.make_preprocessing(
            filter_bs=self.args.filter_bs, per_area=False
        )

        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test, self.area_X_train, self.area_X_val, self.area_X_test, self.area_y_train, self.area_y_val, self.area_y_test, self.exogenous_data_train, self.exogenous_data_val = (
            processing.make_postprocessing(X_train, X_val, X_test, y_train, y_val, y_test, exogenous_data_train,
                                           exogenous_data_val, x_scaler, y_scaler))

        processing.get_input_dims(X_train=self.X_train, exogenous_data_train=self.exogenous_data_train)

        self.input_dim, self.exogenous_dim = processing.get_input_dims(self.X_train, self.exogenous_data_train)

        print(self.input_dim, self.exogenous_dim)

        self.model = self.get_model(model=args.model_name,
                          input_dim=self.input_dim,
                          out_dim=self.y_train.shape[1],
                          lags=args.num_lags,
                          exogenous_dim=self.exogenous_dim,
                          seed=args.seed)



    def fit(self, idxs=[0], log_per=1):
        if self.exogenous_data_train is not None and len(self.exogenous_data_train) > 1:
            exogenous_data_train = self.exogenous_data_train["all"]
            exogenous_data_val = self.exogenous_data_val["all"]
        elif self.exogenous_data_train is not None and len(self.exogenous_data_train) == 1:
            cid = next(iter(self.exogenous_data_train.keys()))
            exogenous_data_train = self.exogenous_data_train[cid]
            exogenous_data_val = self.exogenous_data_val[cid]
        else:
            exogenous_data_train = None
            exogenous_data_val = None
        num_features = len(self.X_train[0][0])

        train_loader = TimeSeriesLoader(X=self.X_train,
                                       y=self.y_train,
                                       num_lags=self.args.num_lags,
                                       num_features=num_features, exogenous_data=exogenous_data_train,
                                       indices=idxs, batch_size=self.args.batch_size, shuffle=False).get_dataloader()

        val_loader = TimeSeriesLoader(X=self.X_val,
                                       y=self.y_val,
                                       num_lags=self.args.num_lags,
                                       num_features=num_features, exogenous_data=exogenous_data_val,
                                       indices=idxs, batch_size=self.args.batch_size, shuffle=False).get_dataloader()

        test_loader = TimeSeriesLoader(X=self.X_test,
                                       y=self.y_test,
                                       num_lags=self.args.num_lags,
                                       num_features=num_features, exogenous_data=exogenous_data_val,
                                       indices=idxs, batch_size=self.args.batch_size, shuffle=False).get_dataloader()
        if not self.args.train_on_flower:
            model = self.train(model=self.model,
                          train_loader=train_loader, test_loader=val_loader,
                          epochs=self.args.epochs,
                          optimizer=self.args.optimizer, lr=self.args.lr,
                          criterion=self.args.criterion,
                          early_stopping=self.args.early_stopping,
                          patience=self.args.patience,
                          plot_history=self.args.plot_history,
                          device=self.args.device, log_per=log_per, cid=self.args.filter_bs)
            return model
        else:
            client = CifarClient(args=self.args, inputdim=self.input_dim,
                                 outdim=self.y_train.shape[1],
                                 exogenous_dim=self.exogenous_dim, trainset=train_loader, valset=val_loader, testset=test_loader, device=self.args.device)

            fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)



