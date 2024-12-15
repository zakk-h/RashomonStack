import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.linalg import inv
from scipy.optimize import minimize
from collections import defaultdict
import json
import os
import sys
sys.setrecursionlimit(5000)

from treefarms import TREEFARMS

class StackedRashomon:
    """
    A class to create a stacked ensemble from the Rashomon set of tree models using TREEFARMS.
    """

    def __init__(
        self,
        treefarms_config=None,
        margin=0.005,
        num_base_models='all',
        split_data=False,
        test_split_ratio=0.2,
        prediction_type='regression',  # 'regression', 'ordinal', 'classification'
        round_predictions=False,
        lambda_reg=1.0,  # Regularization parameter for metamodel
        treefarms_regularization=0.25,  # Separate regularization for TREEFARMS
        treefarms_depth_budget=0,        # Depth budget for TREEFARMS
        random_state=None,
        verbose=True,
        **treefarms_kwargs  # Accept arbitrary TREEFARMS parameters
    ):
        """
        Initialize the StackedRashomon ensemble.

        Parameters:
        - treefarms_config (dict): Configuration dictionary for TREEFARMS. If None, default configurations are used.
        - margin (float): Margin to fetch the Rashomon set (e.g., 0.005 for 0.5%).
        - num_base_models (int or 'all'): Number of base models to select. If 'all', use all models.
        - split_data (bool): Whether to split the training data into layer1 and layer2.
        - test_split_ratio (float): Ratio for splitting data if split_data is True.
        - prediction_type (str): Type of prediction ('regression', 'ordinal', 'classification').
        - round_predictions (bool): Whether to round predictions (used for ordinal).
        - lambda_reg (float): L2 regularization parameter for the metamodel.
        - treefarms_regularization (float): Regularization parameter for TREEFARMS.
        - treefarms_depth_budget (int): Depth budget for TREEFARMS.
        - random_state (int): Random state for reproducibility.
        - verbose (bool): Whether to print verbose messages.
        - **treefarms_kwargs: Arbitrary keyword arguments for TREEFARMS configuration.
        """
        self.margin = margin
        self.num_base_models = num_base_models
        self.split_data = split_data
        self.test_split_ratio = test_split_ratio
        self.prediction_type = prediction_type
        self.round_predictions = round_predictions
        self.lambda_reg = lambda_reg
        self.treefarms_regularization = treefarms_regularization
        self.treefarms_depth_budget = treefarms_depth_budget
        self.random_state = random_state
        self.verbose = verbose

        # TREEFARMS configuration
        self.treefarms_config = self._default_treefarms_config(
            treefarms_regularization=self.treefarms_regularization,
            treefarms_depth_budget=self.treefarms_depth_budget
        )

        # Update TREEFARMS config with any additional kwargs provided
        if treefarms_config is not None:
            self.treefarms_config.update(treefarms_config)
        if treefarms_kwargs:
            self.treefarms_config.update(treefarms_kwargs)

        # Attributes to be set during fitting
        self.base_models = []
        self.selected_models = []
        self.metamodel_weights = None  # Coefficients for metamodel
        self.class_weights = None  # Weights for classification
        self.classes = None  # Array of classes for classification
        self.layer1_X = None
        self.layer1_y = None
        self.layer2_X = None
        self.layer2_y = None

    def _default_treefarms_config(self, treefarms_regularization, treefarms_depth_budget):
        """
        Returns the default TREEFARMS configuration as a dictionary.

        Parameters:
        - treefarms_regularization (float): Regularization parameter for TREEFARMS.
        - treefarms_depth_budget (int): Depth budget for TREEFARMS.

        Returns:
        - default_config (dict): Default configuration for TREEFARMS.
        """
        default_config = {
            "regularization": treefarms_regularization,  # Separate regularization parameter
            "rashomon_bound_multiplier": self.margin,    # Controls the size of the Rashomon set
            "depth_budget": treefarms_depth_budget,      # Depth budget for TREEFARMS
            "time_limit": 0,
            "precision_limit": 0,
            "stack_limit": 0,
            "worker_limit": 1,
            "cancellation": True,
            "look_ahead": True,
            "diagnostics": False,
            "verbose": self.verbose,
            # TBD
        }
        return default_config

    def _select_models(self, models, X, y, num_selected):
        """
        Select a subset of models using a greedy diversity method.

        Parameters:
        - models (list): List of base models.
        - X (pd.DataFrame): Training features.
        - y (pd.Series or np.ndarray): Training targets.
        - num_selected (int): Number of models to select.

        Returns:
        - selected_models (list): List of selected models.
        """
        if num_selected == 'all' or num_selected >= len(models):
            if self.verbose:
                print(f"Selecting all {len(models)} models as num_base_models is 'all' or exceeds available models.")
            return models.copy()

        if self.verbose:
            print(f"Selecting {num_selected} models out of {len(models)} using greedy diversity method.")

        # Get predictions of all models on X
        predictions = np.array([model.predict(X) for model in models])  # Shape: (n_models, n_samples)

        selected_indices = []
        remaining_indices = set(range(len(models)))

        # Initialize by selecting the model with the highest variance
        variances = predictions.var(axis=1)
        first_model = np.argmax(variances)
        selected_indices.append(first_model)
        remaining_indices.remove(first_model)

        while len(selected_indices) < num_selected:
            max_diversity = -np.inf
            next_model = None
            for idx in remaining_indices:
                # Compute diversity as the number of differing predictions with already selected models
                diversity = sum(np.sum(predictions[idx] != predictions[sel_idx]) for sel_idx in selected_indices)
                if diversity > max_diversity:
                    max_diversity = diversity
                    next_model = idx
            if next_model is not None:
                selected_indices.append(next_model)
                remaining_indices.remove(next_model)
                if self.verbose:
                    print(f"Selected model index {next_model} with diversity {max_diversity}.")
            else:
                break  # No more models to select

        selected_models = [models[idx] for idx in selected_indices]
        #print(f"Selected Models: {selected_models}")
        return selected_models

    def _train_metamodel(self, P, y):
        """
        Train the metamodel using the closed-form solution with L2 regularization and coefficients summing to 1.

        Parameters:
        - P (np.ndarray): Predictions from base models (shape: n_samples x n_models).
        - y (np.ndarray): True targets (shape: n_samples,).

        Returns:
        - w (np.ndarray): Metamodel coefficients (shape: n_models,).
        """
        n_samples, n_models = P.shape
        X = P  # Design matrix for metamodel
        y = y.reshape(-1, 1)  # Ensure y is a column vector

        if self.verbose:
            print("Training metamodel using closed-form solution.")

        # Compute A = (X^T X + lambda * I)^-1
        XtX = X.T @ X
        lambda_I = self.lambda_reg * np.eye(n_models)
        A = inv(XtX + lambda_I)

        # Compute A1 and A_Xty
        A1 = A @ np.ones((n_models, 1))  # Shape: (n_models, 1)
        A_Xty = A @ X.T @ y  # Shape: (n_models, 1)

        # Compute numerator and denominator for the adjustment
        numerator = A_Xty - (A1 * (A1.T @ A_Xty - 1))
        denominator = 1 + (A1.T @ A1)
        w = numerator.flatten() / denominator

        # Ensure coefficients sum to 1 (due to numerical precision)
        w /= w.sum()

        if self.verbose:
            print(f"Metamodel coefficients: {w}")

        return w

    def _train_class_weights(self, P, y):
        """
        Train weights for multi-class classification by solving a constrained optimization problem
        for each class separately.

        Parameters:
        - P (np.ndarray): Predictions from base models (shape: n_samples x n_models).
        - y (np.ndarray): True class labels (shape: n_samples,).

        Returns:
        - weights (np.ndarray): Weights for each base model per class (shape: n_models x n_classes).
        - classes (np.ndarray): Array of unique classes.
        """
        n_samples, n_models = P.shape
        classes = np.unique(y)
        n_classes = len(classes)

        if self.verbose:
            print(f"Training classification weights for {n_classes} classes.")

        # One-hot encode the true labels
        y_one_hot = np.zeros((n_samples, n_classes))
        for idx, cls in enumerate(classes):
            y_one_hot[y == cls, idx] = 1

        weights = np.zeros((n_models, n_classes))

        for c in range(n_classes):
            if self.verbose:
                print(f"Optimizing weights for class {classes[c]}.")

            # Define the loss function for class c
            def loss(w):
                return np.sum((P @ w - y_one_hot[:, c]) ** 2) + self.lambda_reg * np.sum(w ** 2)

            # Define the constraint: sum(w) = 1
            constraints = ({
                'type': 'eq',
                'fun': lambda w: np.sum(w) - 1
            })

            # Initial guess: uniform weights
            initial_w = np.ones(n_models) / n_models

            # Bounds: weights can be positive or negative
            bounds = [(-np.inf, np.inf) for _ in range(n_models)]

            if self.verbose:
                print(f"Optimizing weights for class {classes[c]} using SLSQP.")

            # Optimize
            result = minimize(loss, initial_w, method='SLSQP', bounds=bounds, constraints=constraints)

            if not result.success:
                raise ValueError(f"Optimization failed for class {classes[c]}: {result.message}")

            weights[:, c] = result.x

            if self.verbose:
                print(f"Weights for class {classes[c]}: {result.x}")

        return weights, classes
    

    def _validate_tree(self, tree, X, i):
        """
        Validate a tree model from TREEFARMS to ensure it's usable.
        
        Parameters:
        - tree: The tree model to validate
        - X: Sample data for prediction
        - i: Tree index for logging
        
        Returns:
        - bool: Whether the tree is valid
        """
        try:
            if tree is None:
                if self.verbose:
                    print(f"Tree {i} is None")
                return False
                
            # Check if it's a valid tree object
            if not hasattr(tree, 'predict'):
                if self.verbose:
                    print(f"Tree {i} lacks predict method")
                return False
                
            # Try to get a prediction on first row
            sample = X.iloc[[0]]
            pred = tree.predict(sample)
            
            if pred is None:
                if self.verbose:
                    print(f"Tree {i} returned None prediction")
                return False
                
            # Check if prediction shape is valid
            if not isinstance(pred, (np.ndarray, list)) or len(pred) == 0:
                if self.verbose:
                    print(f"Tree {i} returned invalid prediction shape")
                return False
            
            if len(pred) != len(pred):
                return False
                
            if self.verbose:
                print(f"Tree {i} validated successfully - Sample prediction: {pred}")
                
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"Error validating tree {i}: {str(e)}")
            return False

    def _fetch_rashomon_set(self, X, y):
        """
        Fetch the Rashomon set using TREEFARMS.

        Parameters:
        - X (pd.DataFrame or np.ndarray): Training features.
        - y (pd.Series or np.ndarray): Training targets.

        Returns:
        - models (list): List of models in the Rashomon set.
        """
        if self.verbose:
            print("Fetching Rashomon set using TREEFARMS.")

        # Ensure y is a pandas Series with a name attribute
        if isinstance(y, np.ndarray):
            y = pd.Series(y, name="target")
        elif isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]
        elif isinstance(y, pd.Series):
            if y.name is None:
                y = y.copy()
                y.name = "target"
        else:
            y = pd.Series(y, name="target")

        # Ensure X is a pandas DataFrame
        if isinstance(X, np.ndarray):
            # If X is a NumPy array, convert it to DataFrame with generic column names
            X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        elif isinstance(X, pd.Series):
            X = X.to_frame()
        elif not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Initialize TREEFARMS with the configuration
        tf = TREEFARMS(self.treefarms_config)

        # Fit TREEFARMS to get the Rashomon set
        tf.fit(X, y)

        # Retrieve the Rashomon set by iterating over the TREEFARMS model
        models = []
        n_trees = tf.get_tree_count()
        if self.verbose:
            print(f"Number of trees in the Rashomon set: {n_trees}")
        for i in range(n_trees):
            tree = tf[i]
            if self._validate_tree(tree, X, i):
                models.append(tree)

        if self.verbose:
            print(f"Total {len(models)} models retrieved from the Rashomon set.")

        return models

    def fit(self, X, y):
        """
        Fit the StackedRashomon ensemble on the training data.

        Parameters:
        - X (pd.DataFrame or np.ndarray): Training features.
        - y (pd.Series, pd.DataFrame, or np.ndarray): Training targets.
        """
        if self.verbose:
            print("Starting to fit the StackedRashomon ensemble.")

        # Ensure y is a pandas Series with a name attribute
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]
        if isinstance(y, pd.Series):
            if y.name is None:
                y = y.copy()
                y.name = "target"
        elif isinstance(y, np.ndarray):
            y = pd.Series(y, name="target")
        else:
            y = pd.Series(y, name="target")

        # Handle data splitting if required
        if self.split_data:
            if self.verbose:
                print(f"Splitting data into layer1 and layer2 with test_split_ratio={self.test_split_ratio}.")
            X_layer1, X_layer2, y_layer1, y_layer2 = train_test_split(
                X, y, test_size=self.test_split_ratio, random_state=self.random_state, stratify=y if self.prediction_type == 'classification' else None
            )
            self.layer1_X = X_layer1
            self.layer1_y = y_layer1
            self.layer2_X = X_layer2
            self.layer2_y = y_layer2

            # Fetch Rashomon set using layer1 data
            models = self._fetch_rashomon_set(X_layer1, y_layer1)
        else:
            # Fetch Rashomon set using all data
            models = self._fetch_rashomon_set(X, y)

        self.base_models = models

        # Select base models
        if self.num_base_models == 'all':
            selected_models = models.copy()
            if self.verbose:
                print(f"All {len(models)} models selected as base models.")
        else:
            selected_models = self._select_models(
                models,
                self.layer1_X if self.split_data else X,
                self.layer1_y if self.split_data else y,
                self.num_base_models
            )
            if self.verbose:
                print(f"Selected {len(selected_models)} models as base models.")

        self.selected_models = selected_models

        # Prepare data for metamodel training
        if self.split_data:
            if self.verbose:
                print("Generating predictions for metamodel training on layer2 data.")
            # **Do not call fit on the models from Rashomon set as they are already trained**
            # Directly use pre-trained models to predict on layer2 data
            # Ensure layer2_X is a pandas DataFrame
            if isinstance(self.layer2_X, np.ndarray):
                X_layer2_df = pd.DataFrame(self.layer2_X, columns=[f"feature_{i}" for i in range(self.layer2_X.shape[1])])
            elif isinstance(self.layer2_X, pd.Series):
                X_layer2_df = self.layer2_X.to_frame()
            elif isinstance(self.layer2_X, pd.DataFrame):
                X_layer2_df = self.layer2_X
            else:
                X_layer2_df = pd.DataFrame(self.layer2_X)

            P_layer2 = np.array([model.predict(X_layer2_df) for model in self.selected_models]).T  # Shape: (n_samples_layer2, n_models)
            metamodel_X = P_layer2
            metamodel_y = self.layer2_y.values
        else:
            if self.verbose:
                print("Generating predictions for metamodel training on all data.")
            # **Do not call fit on the models from Rashomon set as they are already trained**
            # Directly use pre-trained models to predict on all data
            # Ensure X is a pandas DataFrame
            # print(f"Type of X: {type(X)}")
            # print(f"Sample of X: {X[:5]}")
            if isinstance(X, np.ndarray):
                X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
            elif isinstance(X, pd.Series):
                X_df = X.to_frame()
            elif isinstance(X, pd.DataFrame):
                X_df = X
            else:
                X_df = pd.DataFrame(X)

            # Get predictions from base models on all data
            P = np.array([model.predict(X_df) for model in self.selected_models]).T  # Shape: (n_samples, n_models)
            metamodel_X = P
            metamodel_y = y.values

        # Train metamodel based on prediction type
        if self.prediction_type in ['regression', 'ordinal']:
            if self.verbose:
                print(f"Training metamodel for prediction type '{self.prediction_type}'.")
            self.metamodel_weights = self._train_metamodel(metamodel_X, metamodel_y)
        elif self.prediction_type == 'classification':
            if self.verbose:
                print("Training classification weights.")
            self.class_weights, self.classes = self._train_class_weights(metamodel_X, metamodel_y)
        else:
            raise ValueError("Invalid prediction_type. Choose from 'regression', 'ordinal', 'classification'.")

        if self.verbose:
            print("Finished fitting the StackedRashomon ensemble.")

    def predict(self, X_test):
        """
        Generate predictions on the test set using the stacked ensemble.

        Parameters:
        - X_test (pd.DataFrame or np.ndarray): Test features.

        Returns:
        - predictions (np.ndarray): Predicted values or class labels.
        """
        if self.verbose:
            print("Generating predictions on the test set.")

        # Convert X_test to pandas DataFrame if it's a NumPy array
        if isinstance(X_test, np.ndarray):
            X_test_df = pd.DataFrame(X_test, columns=[f"feature_{i}" for i in range(X_test.shape[1])])
        elif isinstance(X_test, pd.Series):
            X_test_df = X_test.to_frame()
        elif isinstance(X_test, pd.DataFrame):
            X_test_df = X_test
        else:
            X_test_df = pd.DataFrame(X_test)

        # Get base model predictions
        if self.verbose:
            print("Obtaining base model predictions.")
        P_test = np.array([model.predict(X_test_df) for model in self.selected_models]).T  # Shape: (n_samples, n_models)

        if self.prediction_type in ['regression', 'ordinal']:
            # Apply metamodel
            if self.metamodel_weights is None:
                raise ValueError("Metamodel weights are not trained. Please call fit first.")

            # Compute weighted sum
            y_pred = P_test @ self.metamodel_weights

            if self.prediction_type == 'ordinal' and self.round_predictions:
                if self.verbose:
                    print("Rounding predictions for ordinal type.")
                y_pred = np.round(y_pred)

            return y_pred

        elif self.prediction_type == 'classification':
            if self.class_weights is None:
                raise ValueError("Classification weights are not trained. Please call fit first.")

            if self.verbose:
                print("Aggregating weighted votes for classification.")

            # Compute scores for each class
            scores = P_test @ self.class_weights  # Shape: (n_samples, n_classes)

            # Assign class with the highest score
            predicted_indices = np.argmax(scores, axis=1)
            predicted_classes = self.classes[predicted_indices]

            return predicted_classes

        else:
            raise ValueError("Invalid prediction_type. Choose from 'regression', 'ordinal', 'classification'.")

    def get_base_models(self):
        """
        Return the list of selected base models.

        Returns:
        - selected_models (list): List of selected base models.
        """
        return self.selected_models

    def get_metamodel_weights(self):
        """
        Return the metamodel weights.

        Returns:
        - metamodel_weights (np.ndarray): Weights of the metamodel.
        """
        return self.metamodel_weights

    def get_class_weights(self):
        """
        Return the class weights for classification.

        Returns:
        - class_weights (np.ndarray): Weights for each base model.
        """
        return self.class_weights

from sklearn.datasets import fetch_california_housing, load_iris
from sklearn.metrics import mean_squared_error, accuracy_score

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# -------------------------------
# Classification Task using Iris Dataset
# -------------------------------

# Load a classification dataset
data_cls = load_iris()
X_cls, y_cls = data_cls.data, data_cls.target

# Convert X_cls to a pandas DataFrame with appropriate feature names
X_cls = pd.DataFrame(X_cls, columns=data_cls.feature_names)

# Convert y_cls to a pandas Series with a name
y_cls = pd.Series(y_cls, name="target")

# Initialize the StackedRashomon ensemble for classification
ensemble_cls = StackedRashomon(
    margin=0.005,
    num_base_models='all',
    split_data=False,
    test_split_ratio=0.3,
    prediction_type='classification',
    round_predictions=False,  # Not used for classification
    lambda_reg=1.0,
    random_state=42,
)

# Fit the ensemble
ensemble_cls.fit(X_cls, y_cls)

# Make predictions on the training data
y_pred_cls = ensemble_cls.predict(X_cls)

# Evaluate the classification performance on the training data
accuracy = accuracy_score(y_cls, y_pred_cls)
print(f"Classification Accuracy: {accuracy:.4f}")

# -------------------------------
# Regression Task using California Housing Dataset
# -------------------------------

from sklearn.datasets import fetch_california_housing

# Load a regression dataset
data_reg = fetch_california_housing()
X_reg, y_reg = data_reg.data, data_reg.target

# Convert X_reg to a pandas DataFrame with appropriate feature names
X_reg = pd.DataFrame(X_reg, columns=data_reg.feature_names)

# Convert y_reg to a pandas Series with a name
y_reg = pd.Series(y_reg, name="target")

# Initialize the StackedRashomon ensemble for regression
ensemble_reg = StackedRashomon(
    margin=0.005,
    num_base_models=5,
    split_data=True,
    test_split_ratio=0.3,
    prediction_type='regression',
    round_predictions=False,
    lambda_reg=1.0,
    random_state=42,
)

ensemble_reg.fit(X_reg, y_reg)

y_pred_reg = ensemble_reg.predict(X_reg)

mse = mean_squared_error(y_reg, y_pred_reg)
print(f"Regression Mean Squared Error: {mse:.4f}")
