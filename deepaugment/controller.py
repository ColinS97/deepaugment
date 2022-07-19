# (C) 2019 Baris Ozmen <hbaristr@gmail.com>

import skopt
import numpy as np
import aug_lib


MAX_MAGNITUDE = aug_lib.PARAMETER_MAX
AUG_TYPES = []


class Controller:

    opt = None  # used only if method is bayesian optimization
    random_search_space = None  # used only if method is random search

    def __init__(self, config):
        """Initiliaze Controller either as a Bayesian Optimizer or as a Random Search Algorithm

        Args:
             config (dict)
        """
        for transform in aug_lib.ALL_TRANSFORMS:
            AUG_TYPES.append(str(transform))
        if config["method"].startswith("bayes"):
            self.method = "bayesian_optimization"
            self.init_skopt(config["opt_initial_points"])
        else:
            raise ValueError

    def augment_type_chooser():
        """A random function to choose among augmentation types

        Returns:
            function object: np.random.choice function with AUG_TYPES input
        """
        return np.random.choice(AUG_TYPES)

    def init_skopt(self, opt_initial_points):
        """Initialize as scikit-optimize (skopt) Optimizer with a 5-dimensional search space

        Aligned with skopt ask-tell design pattern (https://geekyisawesome.blogspot.com/2018/07/hyperparameter-tuning-using-scikit.html)

        Args:
            opt_initial_points (int): number of random initial points for the optimizer
        """

        self.opt = skopt.Optimizer(
            [
                skopt.space.Categorical(AUG_TYPES, name="A_aug1_type"),
                skopt.space.Integer(0, MAX_MAGNITUDE, name="A_aug1_magnitude"),
                skopt.space.Categorical(AUG_TYPES, name="A_aug2_type"),
                skopt.space.Integer(0, MAX_MAGNITUDE, name="A_aug2_magnitude"),
                skopt.space.Categorical(AUG_TYPES, name="B_aug1_type"),
                skopt.space.Integer(0, MAX_MAGNITUDE, name="B_aug1_magnitude"),
                skopt.space.Categorical(AUG_TYPES, name="B_aug2_type"),
                skopt.space.Integer(0, MAX_MAGNITUDE, name="B_aug2_magnitude"),
                skopt.space.Categorical(AUG_TYPES, name="C_aug1_type"),
                skopt.space.Integer(0, MAX_MAGNITUDE, name="C_aug1_magnitude"),
                skopt.space.Categorical(AUG_TYPES, name="C_aug2_type"),
                skopt.space.Integer(0, MAX_MAGNITUDE, name="C_aug2_magnitude"),
                skopt.space.Categorical(AUG_TYPES, name="D_aug1_type"),
                skopt.space.Integer(0, MAX_MAGNITUDE, name="D_aug1_magnitude"),
                skopt.space.Categorical(AUG_TYPES, name="D_aug2_type"),
                skopt.space.Integer(0, MAX_MAGNITUDE, name="D_aug2_magnitude"),
                skopt.space.Categorical(AUG_TYPES, name="E_aug1_type"),
                skopt.space.Integer(0, MAX_MAGNITUDE, name="E_aug1_magnitude"),
                skopt.space.Categorical(AUG_TYPES, name="E_aug2_type"),
                skopt.space.Integer(0, MAX_MAGNITUDE, name="E_aug2_magnitude"),
            ],
            n_initial_points=opt_initial_points,
            base_estimator="RF",  # Random Forest estimator
            acq_func="EI",  # Expected Improvement
            acq_optimizer="auto",
            random_state=0,
        )

    def init_random_search(self):
        """Initializes random search as the search space is list of random functions"""
        self.random_search_space = [
            self.augment_type_chooser,
            np.random.rand,
            self.augment_type_chooser,
            np.random.rand,
            np.random.rand,
        ]

    def ask(self):
        """Ask controller for the next hyperparameter search.


        If Bayesian Optimizer, samples next hyperparameters by its internal statistic calculations (Random Forest Estimators, Gaussian Processes, etc.). If Random Search, samples randomly
        Based on ask-tell design pattern

        Returns:
            list: list of hyperparameters
        """
        if self.method == "bayesian_optimization":
            return self.opt.ask()

    def tell(self, trial_hyperparams, f_val):
        """Tells the controller result of previous tried hyperparameters

        If Bayesian Optimizer, records this results and updates its internal statistical model. If Random Search does nothing, since it will not affect future (random) samples.

        Args:
            trial_hyperparams (list): list of tried hyperparamters
            f_val (float): trial cost
        """
        if self.method == "bayesian_optimization":
            self.opt.tell(trial_hyperparams, f_val)
