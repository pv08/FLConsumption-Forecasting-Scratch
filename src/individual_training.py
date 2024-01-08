from src.base.trainers import Trainers
from src.dataset.processing import Processsing
from src.data import TimeSeriesLoader


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

        X_train, X_val, y_train, y_val, exogenous_data_train, exogenous_data_val, x_scaler, y_scaler = processing.make_preprocessing(
            filter_bs="LesCorts", per_area=False
        )

        self.X_train, self.X_val, self.y_train, self.y_val, self.area_X_train, self.area_X_val, self.area_y_train, self.area_y_val, self.exogenous_data_train, self.exogenous_data_val = (
            processing.make_postprocessing(X_train, X_val, y_train, y_val, exogenous_data_train,
                                           exogenous_data_val, x_scaler, y_scaler))

        processing.get_input_dims(X_train=self.X_train, exogenous_data_train=self.exogenous_data_train)

        input_dim, exogenous_dim = processing.get_input_dims(self.X_train, self.exogenous_data_train)

        print(input_dim, exogenous_dim)

        self.model = self.get_model(model=args.model_name,
                          input_dim=input_dim,
                          out_dim=self.y_train.shape[1],
                          lags=args.num_lags,
                          exogenous_dim=exogenous_dim,
                          seed=args.seed)



    def fit(self, idxs=[8, 3, 1, 10, 9], log_per=1):
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

        model = self.train(self.model,
                      train_loader, val_loader,
                      epochs=self.args.epochs,
                      optimizer=self.args.optimizer, lr=self.args.lr,
                      criterion=self.args.criterion,
                      early_stopping=self.args.early_stopping,
                      patience=self.args.patience,
                      plot_history=self.args.plot_history,
                      device=self.args.device, log_per=log_per)

        return model

