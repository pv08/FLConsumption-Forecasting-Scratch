import matplotlib.pyplot as plt
import os


def mkdir_if_not_exists(default_save_path: str):
    if not os.path.exists(default_save_path):
        os.mkdir(default_save_path)

def get_params(alg):
    if alg == "fedprox":
        return {"mu": 0.01}
    elif alg == "fednova":
        return {"rho": 0.}
    elif alg == "fedadagrad":
        return {"beta_1": 0., "eta": 0.1, "tau": 1e-2}
    elif alg == "fedyogi":
        return {"beta_1": 0.9, "beta_2": 0.99, "eta": 0.01, "tau": 1e-3}
    elif alg == "fedadam":
        return {"beta_1": 0.9, "beta_2": 0.99, "eta": 0.01, "tau": 1e-3}
    elif alg == "fedavgm":
        return {"server_momentum": 0., "server_lr": 1.}
    else:
        return None

def plot_train_curve(train_history, test_history, title, fig_name):
    plt.title(title)
    plt.plot(train_history, label='Train')
    plt.plot(test_history, label='Test')
    plt.legend()
    plt.show()
    mkdir_if_not_exists('etc/')
    mkdir_if_not_exists('etc/results/')
    plt.savefig(f'etc/results/{fig_name}.png')
    plt.close()