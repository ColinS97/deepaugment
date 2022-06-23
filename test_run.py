from deepaugment.deepaugment import DeepAugment

from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# child_epochs set to 10 for a quick run, but it should be >=50 for a proper analysis
my_config = {"child_epochs": 2, "opt_samples": 1}

# X_train.shape -> (N, M, M, 3)
# y_train.shape -> (N)
deepaug = DeepAugment(images=x_train, labels=y_train, config=my_config)
