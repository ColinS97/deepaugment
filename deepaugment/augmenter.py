# (C) 2019 Baris Ozmen <hbaristr@gmail.com>

import numpy as np
import aug_lib
from PIL import Image
import random


def apply_transform(aug_type, magnitude, img):
    # ich nehme an die daten kommen normalisiert hier an
    propability = 0.5
    X_denormed = img * 255
    denormed_pil_image = Image.fromarray(np.uint8(X_denormed))
    X = denormed_pil_image

    if aug_type == "<identity>":
        X_aug = aug_lib.identity.pil_transformer(propability, magnitude)(X)
    elif aug_type == "<AutoContrast>":
        X_aug = aug_lib.auto_contrast.pil_transformer(propability, magnitude)(X)
    elif aug_type == "<Equalize>":
        X_aug = aug_lib.equalize.pil_transformer(propability, magnitude)(X)
    elif aug_type == "<Rotate>":
        X_aug = aug_lib.rotate.pil_transformer(propability, magnitude)(X)
    elif aug_type == "<Solarize>":
        X_aug = aug_lib.solarize.pil_transformer(propability, magnitude)(X)
    elif aug_type == "<Color>":
        X_aug = aug_lib.color.pil_transformer(propability, magnitude)(X)
    elif aug_type == "<Posterize>":
        X_aug = aug_lib.posterize.pil_transformer(propability, magnitude)(X)
    elif aug_type == "<Contrast>":
        X_aug = aug_lib.contrast.pil_transformer(propability, magnitude)(X)
    elif aug_type == "<Brightness>":
        X_aug = aug_lib.brightness.pil_transformer(propability, magnitude)(X)
    elif aug_type == "<Sharpness>":
        X_aug = aug_lib.sharpness.pil_transformer(propability, magnitude)(X)
    elif aug_type == "<ShearX>":
        X_aug = aug_lib.shear_x.pil_transformer(propability, magnitude)(X)
    elif aug_type == "<ShearY>":
        X_aug = aug_lib.shear_y.pil_transformer(propability, magnitude)(X)
    elif aug_type == "<TranslateX>":
        X_aug = aug_lib.translate_x.pil_transformer(propability, magnitude)(X)
    elif aug_type == "<TranslateY>":
        X_aug = aug_lib.translate_y.pil_transformer(propability, magnitude)(X)
    else:
        raise ValueError
    np_image = np.array(X_aug)
    normed_image = np_image / 255
    return normed_image


def transform(aug_type, magnitude, X):
    # ich nehme an die daten kommen normalisiert hier an
    for index, img in enumerate(X):
        X[index] = apply_transform(aug_type, magnitude, img)
    return X


def augment_by_policy(X, y, *hyperparams):
    augs = []
    for i in range(0, len(hyperparams) - 1, 4):
        augs.append(hyperparams[i : i + 4])

    aug = random.choice(augs)

    X_aug = transform(aug[0], aug[1], X)  # apply first policy

    X_aug = transform(aug[2], aug[3], X_aug)  # apply second policy

    augmented_data = {
        "X_train": X_aug,
        "y_train": y,
    }

    return augmented_data  # augmenteed data is mostly smaller than whole data
