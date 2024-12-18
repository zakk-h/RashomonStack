import numpy as np
import matplotlib.pyplot as plt
import optuna
from copy import deepcopy

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

from ipywidgets import interact, IntSlider
from sklearn.base import clone

# We will define a class similar to before, but we'll add handling of multiple model types in tuning.
class SparseStackedEnsemble:
    def __init__(self, layer1_models, X_train, y_train, X_test, y_test):
        self.layer1_models = layer1_models
        self.original_models = {k: clone(v) for k,v in layer1_models.items()}
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        self.removal_sequence = []
        self.test_mse_sequence = []
        self.weights_sequence = []
        self.current_models = list(self.layer1_models.keys())

    def _compute_weights(self, X, y):
        """
        Compute weights w minimizing ||y - Xw||^2 subject to c^T w = 1.

        We form the KKT conditions:
        [X^T X   c] [w     ] = [X^T y]
        [c^T     0] [lambda]   [  1  ]

        Solve using np.linalg.lstsq which handles singularities.
        """
        n, m = X.shape
        c = np.ones(m)

        XtX = X.T @ X
        Xty = X.T @ y

        # Construct KKT system
        M = np.block([
            [XtX, c.reshape(-1,1)],
            [c.reshape(1,-1), np.zeros((1,1))]
        ])
        rhs = np.concatenate([Xty, [1]])

        # Solve least squares
        sol, residuals, rank, s = np.linalg.lstsq(M, rhs, rcond=None)
        w = sol[:m]
        return w


    def _get_predictions(self, models_subset=None, dataset="train"):
        if models_subset is None:
            models_subset = self.current_models
        if dataset == "train":
            X_data = self.X_train
        else:
            X_data = self.X_test

        preds = []
        for name in models_subset:
            model = self.layer1_models[name]
            pred = model.predict(X_data)
            preds.append(pred)
        return np.column_stack(preds)

    def _evaluate_mse(self, w, X_pred, y):
        ensemble_pred = X_pred @ w
        return mean_squared_error(y, ensemble_pred)

    def _model_removal_criterion(self, train_preds, y):
        # Compute a cost for each model: MSE_j * (1/diversity_j)
        m = train_preds.shape[1]
        mse_list = []
        for j in range(m):
            mse_j = mean_squared_error(y, train_preds[:, j])
            mse_list.append(mse_j)

        diversity_list = []
        for j in range(m):
            diffs = []
            for k in range(m):
                if k != j:
                    diffs.append(np.mean(np.abs(train_preds[:, j] - train_preds[:, k])))
            diversity_j = np.mean(diffs) if diffs else 0.0001
            diversity_list.append(diversity_j)

        cost_list = []
        for j in range(m):
            cost_j = mse_list[j] * (1 / (diversity_list[j] + 1e-12))
            cost_list.append(cost_j)

        return np.argmax(cost_list)

    def fit_full_sequence(self):
        # Assuming models are already fitted
        while len(self.current_models) > 0:
            X_pred_train = self._get_predictions(self.current_models, "train")
            X_pred_test = self._get_predictions(self.current_models, "test")

            if X_pred_train.shape[1] == 1:
                w = np.array([1.0])
            else:
                w = self._compute_weights(X_pred_train, self.y_train)

            mse_test = self._evaluate_mse(w, X_pred_test, self.y_test)
            self.test_mse_sequence.append(mse_test)
            self.weights_sequence.append((self.current_models.copy(), w.copy()))
            self.removal_sequence.append(self.current_models.copy())

            if len(self.current_models) == 1:
                break

            idx_to_remove = self._model_removal_criterion(X_pred_train, self.y_train)
            model_to_remove = self.current_models[idx_to_remove]
            self.current_models.remove(model_to_remove)

    def plot_mse_vs_num_models(self):
        self.num_models_list = [len(m) for m in self.removal_sequence]
        plt.figure(figsize=(8, 5))
        plt.plot(self.num_models_list, self.test_mse_sequence, marker='o')
        plt.xlabel("Number of Models")
        plt.ylabel("MSE on Test Set")
        plt.title("Number of Models vs MSE")
        plt.grid(True)
        plt.show()
        best_num_models_idx = np.argmin(self.test_mse_sequence)
        return self.num_models_list[best_num_models_idx]

    def choose_num_models(self, num_models):
        for (subset, w), mse in zip(self.weights_sequence, self.test_mse_sequence):
            if len(subset) == num_models:
                self.final_models = subset
                self.final_mse = mse
                self.final_weights = w
                break

    def tune_hyperparameters(self, n_trials=20):
        tuned_models = {}
        # Tune each model type according to what it is
        for model_name in self.final_models:
            original_model = self.original_models[model_name]
            # Check model type and tune accordingly
            if isinstance(original_model, RandomForestRegressor):
                # tune n_estimators (up to 1000), max_depth (up to 20)
                def objective(trial):
                    n_estimators = trial.suggest_int("n_estimators", 50, 1000)
                    max_depth = trial.suggest_int("max_depth", 2, 20)
                    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
                    m = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                              min_samples_split=min_samples_split, random_state=42)
                    m.fit(self.X_train, self.y_train)
                    preds = m.predict(self.X_test)
                    return mean_squared_error(self.y_test, preds)

            elif isinstance(original_model, GradientBoostingRegressor):
                def objective(trial):
                    n_estimators = trial.suggest_int("n_estimators", 50, 1000)
                    max_depth = trial.suggest_int("max_depth", 2, 20)
                    learning_rate = trial.suggest_float("learning_rate", 0.001, 0.1, log=True)
                    m = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                                  learning_rate=learning_rate, random_state=42)
                    m.fit(self.X_train, self.y_train)
                    preds = m.predict(self.X_test)
                    return mean_squared_error(self.y_test, preds)

            elif isinstance(original_model, XGBRegressor):
                def objective(trial):
                    n_estimators = trial.suggest_int("n_estimators", 50, 1000)
                    max_depth = trial.suggest_int("max_depth", 2, 20)
                    learning_rate = trial.suggest_float("learning_rate", 0.001, 0.1, log=True)
                    m = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                     learning_rate=learning_rate, random_state=42,
                                     verbosity=0)
                    m.fit(self.X_train, self.y_train)
                    preds = m.predict(self.X_test)
                    return mean_squared_error(self.y_test, preds)

            elif isinstance(original_model, Lasso):
                def objective(trial):
                    alpha = trial.suggest_float("alpha", 0.0001, 10.0, log=True)
                    m = Lasso(alpha=alpha, max_iter=10000, random_state=42)
                    m.fit(self.X_train, self.y_train)
                    preds = m.predict(self.X_test)
                    return mean_squared_error(self.y_test, preds)

            elif isinstance(original_model, Ridge):
                def objective(trial):
                    alpha = trial.suggest_float("alpha", 0.0001, 10.0, log=True)
                    m = Ridge(alpha=alpha, random_state=42)
                    m.fit(self.X_train, self.y_train)
                    preds = m.predict(self.X_test)
                    return mean_squared_error(self.y_test, preds)

            elif isinstance(original_model, ElasticNet):
                def objective(trial):
                    alpha = trial.suggest_float("alpha", 0.0001, 10.0, log=True)
                    l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
                    m = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000, random_state=42)
                    m.fit(self.X_train, self.y_train)
                    preds = m.predict(self.X_test)
                    return mean_squared_error(self.y_test, preds)

            elif isinstance(original_model, KNeighborsRegressor):
                def objective(trial):
                    n_neighbors = trial.suggest_int("n_neighbors", 1, 50)
                    weights = trial.suggest_categorical("weights", ["uniform", "distance"])
                    p = trial.suggest_int("p", 1, 2) # Minkowski distance
                    m = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, p=p)
                    m.fit(self.X_train, self.y_train)
                    preds = m.predict(self.X_test)
                    return mean_squared_error(self.y_test, preds)

            else:
                # For linear regression or models with no tunable parameters
                # just skip tuning
                tuned_models[model_name] = self.layer1_models[model_name]
                continue

            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=n_trials)
            best_params = study.best_params

            # Rebuild model with best params
            m = clone(original_model)
            for param, val in best_params.items():
                setattr(m, param, val)
            m.fit(self.X_train, self.y_train)
            tuned_models[model_name] = m

        self.final_tuned_models = tuned_models
        X_pred_test = self._get_predictions(self.final_tuned_models.keys(), "test")
        final_ensemble_pred = X_pred_test @ self.final_weights
        self.final_tuned_mse = mean_squared_error(self.y_test, final_ensemble_pred)

    def get_results(self):
        return {
            "removal_sequence": self.removal_sequence,
            "test_mse_sequence": self.test_mse_sequence,
            "weights_sequence": self.weights_sequence,
            "final_models": getattr(self, 'final_models', None),
            "final_mse": getattr(self, 'final_mse', None),
            "final_tuned_models": getattr(self, 'final_tuned_models', None),
            "final_tuned_mse": getattr(self, 'final_tuned_mse', None)
        }


####################################
# Example usage with California Housing
####################################
data = fetch_california_housing()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define 40 models:
layer1_models = {}

# 5 Linear (1 LR, 2 Ridge, 1 Lasso, 1 ENet)
layer1_models["linear_1"] = LinearRegression()
layer1_models["ridge_1"] = Ridge(alpha=1.0, random_state=42)
layer1_models["ridge_2"] = Ridge(alpha=10.0, random_state=42)
layer1_models["lasso_1"] = Lasso(alpha=0.1, random_state=42, max_iter=10000)
layer1_models["enet_1"] = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=10000)

# 5 KNN with different neighbors
for i, n in enumerate([3,5,10,20,30]):
    layer1_models[f"knn_{i}"] = KNeighborsRegressor(n_neighbors=n)

# 5 RandomForest variants
rf_params = [
    (5), (10), (15), (20), (25)
]
for i, (depth) in enumerate(rf_params):
    layer1_models[f"rf_{i}"] = RandomForestRegressor(n_estimators=1000, max_depth=depth, random_state=42)

# 5 GradientBoosting variants
gbr_params = [
    (2), (4), (6), (10), (12)
]
for i, (depth) in enumerate(gbr_params):
    layer1_models[f"gbr_{i}"] = GradientBoostingRegressor(n_estimators=1000, max_depth=depth, random_state=42)

# 5 XGB variants
xgb_params = [
    (2,0.1), (4,0.05), (6,0.05), (8,0.01), (10,0.01)
]
for i,(depth, lr) in enumerate(xgb_params):
    layer1_models[f"xgb_{i}"] = XGBRegressor(n_estimators=1000, max_depth=depth, learning_rate=lr, random_state=42, verbosity=0)

# 5 Lasso variants (different alphas)
for i, alpha in enumerate([0.001,0.01,0.1,1,10]):
    layer1_models[f"lasso_{i+2}"] = Lasso(alpha=alpha, max_iter=10000, random_state=42)

# 5 Ridge variants (different alphas)
for i, alpha in enumerate([0.001,0.01,0.1,1,10]):
    layer1_models[f"ridge_{i+3}"] = Ridge(alpha=alpha, random_state=42)


# Fit all models
for name, model in layer1_models.items():
    model.fit(X_train, y_train)

ensemble = SparseStackedEnsemble(layer1_models, X_train, y_train, X_test, y_test)
ensemble.fit_full_sequence()
optimal_num_models = ensemble.plot_mse_vs_num_models()

ensemble.choose_num_models(optimal_num_models)

# Tune hyperparams (this may take time)
ensemble.tune_hyperparameters(n_trials=5)

results = ensemble.get_results()
print("Final chosen models:", results["final_models"])
print("Final MSE before tuning:", results["final_mse"])
print("Final MSE after tuning:", results["final_tuned_mse"])
