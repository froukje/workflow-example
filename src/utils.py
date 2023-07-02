import random
import pandas as pd
import numpy as np

RANDOM_STATE = 42


def set_seed(SEED=RANDOM_STATE):
    random.seed(SEED)
    np.random.seed(SEED)


def upsample(X_train, y_train):
    df_tmp = pd.concat([X_train, y_train], axis=1, join="inner")
    df_pos = df_tmp[df_tmp["Potability"] == 1]
    df_neg = df_tmp[df_tmp["Potability"] == 0]

    df_pos_upsample = resample(
        df_pos,
        replace=True,
        n_samples=(df_tmp["Potability"] == 0).sum(),
        random_state=RANDOM_STATE,
    )
    df_upsampled = pd.concat([df_pos_upsample, df_neg])
    df_upsampled = df_upsampled.sample(frac=1)
    X_train = df_upsampled.iloc[:, :-1]
    y_train = df_upsampled["Potability"]
    return X_train, y_train


def print_feature_importances(feature_names, importances):
    feats = {}
    for feature, importance in zip(feature_names, importances):
        feats[feature] = importance
    feats = sorted(feats.items(), key=lambda x: x[1], reverse=True)
    for feat in feats:
        print(f"{feat[0]}: {feat[1]:.3f}")
