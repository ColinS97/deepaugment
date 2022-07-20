# from deepaugment.deepaugment import DeepAugment
# import numpy as np
# from pathlib import Path
# import datetime
# import keras

# path = "pathmnist.npz"
# x_train = None
# y_train = None
# with np.load(path) as data:
#     lst = data.files
#     x_train = data["train_images"]
#     y_train = data["train_labels"]


# now = datetime.datetime.now()
# EXPERIMENT_NAME = f"{now.month:02}-{now.day:02}_{now.hour:02}-{now.minute:02}"
# # child_epochs set to 10 for a quick run, but it should be >=50 for a proper analysis

# nb_path = "logs_" + path + EXPERIMENT_NAME

# Path(nb_path).mkdir(parents=True, exist_ok=True)
# my_config = {
#     "model": "basiccnn",
#     "method": "bayesian_optimization",
#     "opt_samples": 3,
#     "opt_last_n_epochs": 3,
#     "opt_initial_points": 100,
#     "child_epochs": 120,
#     "child_first_train_epochs": 0,
#     "child_batch_size": 64,
#     "notebook_path": nb_path + "/notebook.csv",
# }

# # X_train.shape -> (N, M, M, 3)
# # y_train.shape -> (N)
# deepaug = DeepAugment(images=x_train, labels=y_train, config=my_config)
from deepaugment.deepaugment import DeepAugment
from keras.datasets import fashion_mnist
import datetime
from pathlib import Path
import numpy as np
import ssl

# This restores the same behavior as before.

ssl._create_default_https_context = ssl._create_unverified_context

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

now = datetime.datetime.now()
EXPERIMENT_NAME = f"{now.month:02}-{now.day:02}_{now.hour:02}-{now.minute:02}"
# child_epochs set to 10 for a quick run, but it should be >=50 for a proper analysis

path = "fashion_mnist"
nb_path = "logs_" + path + EXPERIMENT_NAME
Path(nb_path).mkdir(parents=True, exist_ok=True)
# child_epochs set to 10 for a quick run, but it should be >=50 for a proper analysis
my_config = {
    "model": "basiccnn",
    "method": "bayesian_optimization",
    "opt_samples": 2,
    "opt_last_n_epochs": 3,
    "opt_initial_points": 10,
    "child_epochs": 100,
    "child_first_train_epochs": 25,
    "child_batch_size": 64,
    "notebook_path": nb_path + "/notebook.csv",
}


# X_train.shape -> (N, M, M, 3)
# y_train.shape -> (N)

stacked_img = np.stack((x_train,) * 3, axis=-1)
print(stacked_img.shape)
deepaug = DeepAugment(images=stacked_img, labels=y_train, config=my_config)

best_policies = deepaug.optimize(100)

print(best_policies)
