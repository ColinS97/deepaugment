from tensorflow import keras
import numpy as np
import pandas as pd


import sys
from os.path import dirname, realpath

file_path = realpath(__file__)
dir_of_file = dirname(file_path)
sys.path.insert(0, dir_of_file)

from augmenter import augment_by_policy


def deepaugment_image_generator(X, y, policy, batch_size=64, augment_chance=1):
    """Yields batch of images after applying random augmentations from the policy

    Each image is augmented by 100% chance. If augmented, one of the augment-chain in the policy is applied.
    Which augment-chain to apply is chosen randomly.

    Args:
        X (numpy.array): numpy array of images
        labels (numpy.array): numpy array of labels
        policy (string/ pandas df): policy to apply to images

    Returns:
    """
    if type(policy) == str:
        policy_df = pd.read_csv(policy)
        policy_df = policy_df[
            ["aug1_type", "aug1_magnitude", "aug2_type", "aug2_magnitude"]
        ]
        policy = policy_df.to_dict(orient="records")

    print("Policies are:")
    print(policy)
    print()

    while True:
        ix = np.arange(len(X))
        np.random.shuffle(ix)
        for i in range(len(X) // batch_size):
            _ix = ix[i * batch_size : (i + 1) * batch_size]
            _X = X[_ix]
            _y = y[_ix]

            tiny_batch_size = 4
            aug_X = _X[0:tiny_batch_size]
            aug_y = _y[0:tiny_batch_size]
            for j in range(1, len(_X) // tiny_batch_size):
                tiny_X = _X[j * tiny_batch_size : (j + 1) * tiny_batch_size]
                tiny_y = _y[j * tiny_batch_size : (j + 1) * tiny_batch_size]
                if np.random.rand() <= augment_chance:
                    aug_chain = np.random.choice(policy)
                    aug_chain[
                        "portion"
                    ] = 1.0  # last element is portion, which we want to be 1
                    hyperparams = list(aug_chain.values())

                    aug_data = augment_by_policy(tiny_X, tiny_y, *hyperparams)

                    aug_X = np.concatenate([aug_X, aug_data["X_train"]])
                    aug_y = np.concatenate([aug_y, aug_data["y_train"]])
                else:
                    aug_X = np.concatenate([aug_X, tiny_X])
                    aug_y = np.concatenate([aug_y, tiny_y])
            yield aug_X, aug_y


def load_k_policies_from_csv(nb_df, k=20):
    """loading the top policies from dataframe into parsable format for image generator
    Args:
        nb_df (pd.DF):
        k (int, optional): Defaults to 20.
    Returns:
        list: list of dictionaries
    """
    trial_avg_val_acc_df = (
        nb_df.drop_duplicates(["trial_no", "sample_no"])
        .groupby("trial_no")
        .mean()["mean_late_val_acc"]
        .reset_index()
    )[["trial_no", "mean_late_val_acc"]]

    x_df = pd.merge(
        nb_df.drop(columns=["mean_late_val_acc"]),
        trial_avg_val_acc_df,
        on="trial_no",
        how="left",
    )

    x_df = x_df.sort_values("mean_late_val_acc", ascending=False)

    baseline_val_acc = x_df[x_df["trial_no"] == 0]["mean_late_val_acc"].values[0]

    x_df["expected_accuracy_increase(%)"] = (
        x_df["mean_late_val_acc"] - baseline_val_acc
    ) * 100

    top_df = x_df.drop_duplicates(["trial_no"]).sort_values(
        "mean_late_val_acc", ascending=False
    )[:k]

    SELECT = [
        "trial_no",
        "A_aug1_type",
        "A_aug1_magnitude",
        "A_aug2_type",
        "A_aug2_magnitude",
        "B_aug1_type",
        "B_aug1_magnitude",
        "B_aug2_type",
        "B_aug2_magnitude",
        "C_aug1_type",
        "C_aug1_magnitude",
        "C_aug2_type",
        "C_aug2_magnitude",
        "D_aug1_type",
        "D_aug1_magnitude",
        "D_aug2_type",
        "D_aug2_magnitude",
        "E_aug1_type",
        "E_aug1_magnitude",
        "E_aug2_type",
        "E_aug2_magnitude",
        "mean_late_val_acc",
        "expected_accuracy_increase(%)",
    ]
    top_df = top_df[SELECT]

    top_policies_list = top_df[
        [
            "A_aug1_type",
            "A_aug1_magnitude",
            "A_aug2_type",
            "A_aug2_magnitude",
            "B_aug1_type",
            "B_aug1_magnitude",
            "B_aug2_type",
            "B_aug2_magnitude",
            "C_aug1_type",
            "C_aug1_magnitude",
            "C_aug2_type",
            "C_aug2_magnitude",
            "D_aug1_type",
            "D_aug1_magnitude",
            "D_aug2_type",
            "D_aug2_magnitude",
            "E_aug1_type",
            "E_aug1_magnitude",
            "E_aug2_type",
            "E_aug2_magnitude",
        ]
    ].to_dict(orient="records")

    return top_policies_list


def load_top_df_from_csv(nb_df, k=20):
    """loading the top policies from dataframe into parsable format for image generator
    Args:
        nb_df (pd.DF):
        k (int, optional): Defaults to 20.
    Returns:
        list: list of dictionaries
    """
    trial_avg_val_acc_df = (
        nb_df.drop_duplicates(["trial_no", "sample_no"])
        .groupby("trial_no")
        .mean()["mean_late_val_acc"]
        .reset_index()
    )[["trial_no", "mean_late_val_acc"]]

    x_df = pd.merge(
        nb_df.drop(columns=["mean_late_val_acc"]),
        trial_avg_val_acc_df,
        on="trial_no",
        how="left",
    )

    x_df = x_df.sort_values("mean_late_val_acc", ascending=False)

    baseline_val_acc = x_df[x_df["trial_no"] == 0]["mean_late_val_acc"].values[0]

    x_df["expected_accuracy_increase(%)"] = (
        x_df["mean_late_val_acc"] - baseline_val_acc
    ) * 100

    top_df = x_df.drop_duplicates(["trial_no"]).sort_values(
        "mean_late_val_acc", ascending=False
    )[:k]

    SELECT = [
        "trial_no",
        "A_aug1_type",
        "A_aug1_magnitude",
        "A_aug2_type",
        "A_aug2_magnitude",
        "B_aug1_type",
        "B_aug1_magnitude",
        "B_aug2_type",
        "B_aug2_magnitude",
        "C_aug1_type",
        "C_aug1_magnitude",
        "C_aug2_type",
        "C_aug2_magnitude",
        "D_aug1_type",
        "D_aug1_magnitude",
        "D_aug2_type",
        "D_aug2_magnitude",
        "E_aug1_type",
        "E_aug1_magnitude",
        "E_aug2_type",
        "E_aug2_magnitude",
        "mean_late_val_acc",
        "expected_accuracy_increase(%)",
    ]
    top_df = top_df[SELECT]

    return top_df


def test_deepaugment_image_generator():
    X = np.random.rand(200, 32, 32, 3)

    y = np.random.randint(10, size=200)
    y = keras.utils.np_utils.to_categorical(y)

    batch_size = 64

    policy = [
        {
            "aug1_type": "<rotate>",
            "aug1_magnitude": 5,
            "aug2_type": "<rotate>",
            "aug2_magnitude": 2,
            "aug3_type": "<rotate>",
            "aug3_magnitude": 2,
            "portion": 0.5,
        },
        {
            "aug1_type": "<rotate>",
            "aug1_magnitude": 5,
            "aug2_type": "<rotate>",
            "aug2_magnitude": 2,
            "aug3_type": "<rotate>",
            "aug3_magnitude": 2,
            "portion": 0.2,
        },
    ]

    gen = deepaugment_image_generator(X, y, policy, batch_size=batch_size)

    a = next(gen)
    b = next(gen)
    c = next(gen)
    # if no error happened during next()'s, it is good


if __name__ == "__main__":
    test_deepaugment_generator()
