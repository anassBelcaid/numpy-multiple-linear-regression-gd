"""
NumPy Multiple Linear Regression GD

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - shuffle_xy
def shuffle_xy(X, y, seed=42):
    """Randomly permute feature rows and targets together.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Feature matrix.
    y : np.ndarray, shape (n,)
        Target vector.
    seed : int, optional
        RNG seed for reproducibility (default 42).

    Returns
    -------
    X_shuffled : np.ndarray, shape (n, d)
    y_shuffled : np.ndarray, shape (n,)
    """
    # TODO: Return (X, y) under one shared seeded row permutation
    rng = np.random.default_rng(seed=seed)

    n = X.shape[0]
    idxs = rng.permutation(n)

    return X[idxs], y[idxs]

# Step 2 - split_train_val_test
def split_train_val_test(X, y, train_frac=0.6, val_frac=0.2):
    """Split already-shuffled arrays into contiguous train, val, and test sets."""
    n = X.shape[0]
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    val_end = n_train + n_val

    X_train = X[:n_train]
    y_train = y[:n_train]
    X_val = X[n_train:val_end]
    y_val = y[n_train:val_end]
    X_test = X[val_end:]
    y_test = y[val_end:]

    return X_train, y_train, X_val, y_val, X_test, y_test

# Step 3 - compute_feature_stats (not yet solved)
# TODO: implement

# Step 4 - standardize_features (not yet solved)
# TODO: implement

# Step 5 - add_bias_column (not yet solved)
# TODO: implement

# Step 6 - prepare_design_matrix (not yet solved)
# TODO: implement

# Step 7 - predict_linear (not yet solved)
# TODO: implement

# Step 8 - mse_loss (not yet solved)
# TODO: implement

# Step 9 - mse_gradient (not yet solved)
# TODO: implement

# Step 10 - normal_equation (not yet solved)
# TODO: implement

# Step 11 - initialize_weights (not yet solved)
# TODO: implement

# Step 12 - gd_step (not yet solved)
# TODO: implement

# Step 13 - epoch_train_val_losses (not yet solved)
# TODO: implement

# Step 14 - update_early_stop_state (not yet solved)
# TODO: implement

# Step 15 - init_training_state (not yet solved)
# TODO: implement

# Step 16 - run_one_epoch (not yet solved)
# TODO: implement

# Step 17 - train_batch_gd (not yet solved)
# TODO: implement

# Step 18 - mean_absolute_error (not yet solved)
# TODO: implement

# Step 19 - root_mean_squared_error (not yet solved)
# TODO: implement

# Step 20 - r_squared (not yet solved)
# TODO: implement

# Step 21 - evaluate_regression (not yet solved)
# TODO: implement

# Step 22 - learning_curve_data (not yet solved)
# TODO: implement

# Step 23 - weights_l2_distance (not yet solved)
# TODO: implement

# Step 24 - create_lr_model (not yet solved)
# TODO: implement

# Step 25 - fit_lr_model (not yet solved)
# TODO: implement

# Step 26 - predict_lr_model (not yet solved)
# TODO: implement

# Step 27 - score_lr_model (not yet solved)
# TODO: implement

# Step 28 - compare_with_normal_equation (not yet solved)
# TODO: implement

