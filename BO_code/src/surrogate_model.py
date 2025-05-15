from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import numpy as np
import optuna
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.constraints import Interval
from botorch.models.transforms import Standardize
from fastkan import FastKAN as FastKAN
from kan import KAN
from kan.utils import create_dataset_from_data

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  ## CUDA not applicatble yet
device = torch.device('cpu')

import warnings
import logging

# 关闭警告信息
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

class RegressionToClassificationWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, regressor):
        self.regressor = regressor

    def fit(self, X, y):
        self.regressor.fit(X, y)
        return self

    def predict(self, X):
        return (self.regressor.predict(X) > 0.5).astype(int)

    def predict_proba(self, X):
        preds = self.regressor.predict(X)
        return np.vstack((1 - preds, preds)).T

class SurrogateModel:
    def __init__(self, model_name=None, params=None):
        self.model_name = model_name
        self.params = params if params else {}
        self.model = self._initialize_model()

    def _initialize_model(self):
        models = {
            'LinearRegression': LinearRegression,
            'Ridge': Ridge,
            'Lasso': Lasso,
            'ElasticNet': ElasticNet,
            'KNeighborsRegressor': KNeighborsRegressor,
            'DecisionTreeRegressor': DecisionTreeRegressor,
            'RandomForest': RandomForestRegressor,
            'SVR': SVR,
            'MLPRegressor': MLPRegressor,
            'GradientBoostingRegressor': GradientBoostingRegressor,
            'AdaBoostRegressor': AdaBoostRegressor,
            'ExtraTreesRegressor': ExtraTreesRegressor,
            'XGBoost': xgb.XGBRegressor,
            'LightGBM': lambda **kwargs: lgb.LGBMRegressor(verbose=-1, **kwargs),
            'GaussianProcess': self._initialize_gp_model,
            'KAN': self._initialize_kan_model,
            'FastKAN': self._initialize_fast_kan_model,
        }
        if self.model_name is None or self.model_name not in models.keys():
            raise ValueError(f"Unknown model name: {self.model_name}")
        model_class = models.get(self.model_name)

        return model_class(**self.params)

    def _initialize_gp_model(self):
        return None  # Placeholder for the Gaussian process model

    def _initialize_kan_model(self, **kwargs):

        # Dynamically configure width based on input/output dimensions
        feature_dim = self.params.get("feature_dim")  # Placeholder, dynamically assigned later
        target_dim = self.params.get("target_dim")  # Placeholder, dynamically assigned later
        hidden_layers = self.params.get("hidden_layers")  # Default: 2 hidden layers with 2 nodes each
        
        width = [feature_dim] + hidden_layers + [target_dim]
        
        # Only keep the parameters that KAN explicitly requires
        kan_params = {
            "width": width,
            "grid": self.params.get("grid"),
            "k": self.params.get("k"),
            "mult_arity": self.params.get("mult_arity", 2),
            "noise_scale": self.params.get("noise_scale", 0.3),
            "base_fun": self.params.get("base_fun", 'silu'),
            "symbolic_enabled": self.params.get("symbolic_enabled", True),
            "affine_trainable": self.params.get("affine_trainable", False),
            "grid_eps": self.params.get("grid_eps", 0.02),
            "grid_range": self.params.get("grid_range", [-1, 1]),
            "sp_trainable": self.params.get("sp_trainable", True),
            "sb_trainable": self.params.get("sb_trainable", True),
            # "seed": self.params.get("seed", 1),
            "save_act": self.params.get("save_act", True),
            "sparse_init": self.params.get("sparse_init", False),
            "auto_save": self.params.get("auto_save", False),
            "ckpt_path": self.params.get("ckpt_path", './model'),
            # "state_id": self.params.get("state_id", 0),
            # "round": self.params.get("round", 0),
            "device": device
        }

        return KAN(**kan_params).to(device)
    
    def _initialize_fast_kan_model(self, **kwargs):
        
        # Dynamically configure layers_hidden based on input/output dimensions
        feature_dim = self.params.get("feature_dim")  # Placeholder, dynamically assigned later
        target_dim = self.params.get("target_dim")  # Placeholder, dynamically assigned later
        hidden_layers = self.params.get("hidden_layers")  # Default: 2 hidden layers with 2 nodes each
    
        width = [feature_dim] + hidden_layers + [target_dim]

        kan_params = {
            "layers_hidden": width,
            "grid_min": self.params.get("grid_min", -2.0),
            "grid_max": self.params.get("grid_max", 2.0),
            "num_grids": self.params.get("num_grids"),
            "use_base_update": self.params.get("use_base_update", True),
            "base_activation": self.params.get("base_activation", torch.nn.functional.silu),
            "spline_weight_init_scale": self.params.get("spline_weight_init_scale", 0.1),
        }
    
        return FastKAN(**kan_params).to(device)

    def fit(self, X, y):
        ### MaternKernel as an example, people can add more in the future follow this way
        if self.model_name == 'GaussianProcess':
            covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=X.shape[1], lengthscale_constraint=Interval(0.1, 4.0)))
            outcome_transform = Standardize(m=1)
            self.model = SingleTaskGP(torch.tensor(X, dtype=torch.float32, device=device), torch.tensor(y, dtype=torch.float32, device=device).unsqueeze(-1), outcome_transform=outcome_transform, covar_module=covar_module)
            mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
            fit_gpytorch_mll(mll)
        elif self.model_name in ['KAN', 'FastKAN']:
            self.params["feature_dim"] = X.shape[1]
            self.params["target_dim"] = 1 if len(y.shape) == 1 else y.shape[1]
            print(f"KAN params: {self.params}")
            if self.model_name == 'KAN':
                # self.model = self._initialize_kan_model(self.params)
                X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
                y_tensor = torch.tensor(y, dtype=torch.float32, device=device).unsqueeze(-1)
                dataset = create_dataset_from_data(X_tensor, y_tensor)
                self.model.fit(dataset, opt="LBFGS", steps=self.params.get("steps", 50), lamb=self.params.get("lamb", 0.002), lamb_entropy=self.params.get("lamb_entropy", 2.0))
            elif self.model_name == 'FastKAN':
                # self.model = self._initialize_fast_kan_model(self.params)
                X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
                y_tensor = torch.tensor(y, dtype=torch.float32, device=device).unsqueeze(-1)
                optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params.get("lr"))
                self.model.to(device)
                loss_fn = torch.nn.MSELoss()
                for step in range(self.params.get("steps", 3000)):
                    optimizer.zero_grad()
                    predictions = self.model(X_tensor)
                    loss = loss_fn(predictions, y_tensor)
                    loss.backward()
                    optimizer.step()
        else:
            self.model.fit(X, y)

    def predict(self, X):
        ### MaternKernel as an example, people can add more in the future follow this way
        if self.model_name == 'GaussianProcess':
            model_pred = self.model.posterior(torch.tensor(X, dtype=torch.float32))
            mean = model_pred.mean.detach().cpu().numpy().reshape(-1)
            # std = torch.sqrt(model_pred.variance).detach().cpu().numpy().reshape(-1)
            return mean
        elif self.model_name == 'KAN':
            return self.model(torch.tensor(X, dtype=torch.float32, device=device)).detach().cpu().numpy().flatten()
        elif self.model_name == 'FastKAN':
            with torch.no_grad():
                return self.model(torch.tensor(X, dtype=torch.float32, device=device)).cpu().numpy().flatten()
        else:
            return self.model.predict(X)

    def predict_proba(self, X):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        preds = self.model.predict(X)
        return np.vstack((1 - preds, preds)).T

def hyperparameter_optimization(model_name, X_train, y_train, cls=False, n_trials=20):
    
    feature_dim = X_train.shape[1]
    target_dim = 1 if len(y_train.shape) == 1 else y_train.shape[1]
    
    def objective(trial):
        params = {}
        if model_name == 'RandomForest':
            params['n_estimators'] = trial.suggest_int('n_estimators', 10, 300)
            params['max_depth'] = trial.suggest_int('max_depth', 1, 64)
        elif model_name == 'SVR':
            params['C'] = trial.suggest_loguniform('C', 1e-5, 1e4)
            params['epsilon'] = trial.suggest_loguniform('epsilon', 1e-5, 1e2)
        elif model_name == 'MLPRegressor':
            params['hidden_layer_sizes'] = trial.suggest_categorical('hidden_layer_sizes', [(50,), (100,), (200,), (50,50), (100,100)])
            params['activation'] = trial.suggest_categorical('activation', ['tanh', 'relu'])
            params['solver'] = trial.suggest_categorical('solver', ['adam', 'sgd'])
            params['alpha'] = trial.suggest_loguniform('alpha', 1e-5, 1e-1)
        elif model_name in ['XGBoost', 'LightGBM']:
            params['n_estimators'] = trial.suggest_int('n_estimators', 10, 300)
            params['max_depth'] = trial.suggest_int('max_depth', 1, 64)
            params['learning_rate'] = trial.suggest_loguniform('learning_rate', 1e-4, 1.0)
            if model_name == 'LightGBM':
                params['min_child_samples'] = trial.suggest_int('min_child_samples', 5, 200)
        elif model_name in ['Ridge', 'Lasso', 'ElasticNet']:
            params['alpha'] = trial.suggest_loguniform('alpha', 1e-5, 1e3)
            if model_name == 'ElasticNet':
                params['l1_ratio'] = trial.suggest_uniform('l1_ratio', 0, 1)
        elif model_name == 'KNeighborsRegressor':
            params['n_neighbors'] = trial.suggest_int('n_neighbors', 1, min(len(X_train), 50))
        elif model_name == 'DecisionTreeRegressor':
            params['max_depth'] = trial.suggest_int('max_depth', 1, 64)
        elif model_name in ['GradientBoostingRegressor', 'AdaBoostRegressor', 'ExtraTreesRegressor']:
            params['n_estimators'] = trial.suggest_int('n_estimators', 10, 300)
            if model_name == 'GradientBoostingRegressor' or model_name == 'AdaBoostRegressor':
                params['learning_rate'] = trial.suggest_loguniform('learning_rate', 1e-4, 1.0)
            if model_name == 'GradientBoostingRegressor' or model_name == 'ExtraTreesRegressor':
                params['max_depth'] = trial.suggest_int('max_depth', 1, 64)
        elif model_name == 'KAN':
            params["feature_dim"] = trial.suggest_int("feature_dim", feature_dim, feature_dim)
            params["target_dim"] = trial.suggest_int("target_dim", target_dim, target_dim)
            params['hidden_layers'] = trial.suggest_categorical('hidden_layers', [[2], [4], [8], [2, 2], [4, 4]])
            params['grid'] = trial.suggest_int('grid', 3, 10)
            params['k'] = trial.suggest_int('k', 2, 3)
            # params['noise_scale'] = trial.suggest_loguniform('noise_scale', 1e-2, 0.5)
            params['lamb'] = trial.suggest_loguniform('lamb', 1e-4, 5e-2)
            # params['lamb_entropy'] = trial.suggest_loguniform('lamb_entropy', 1.0, 2.0)
            # params['steps'] = trial.suggest_int('steps', 20, 200)
        elif model_name == 'FastKAN':
            params["feature_dim"] = trial.suggest_int("feature_dim", feature_dim, feature_dim)
            params["target_dim"] = trial.suggest_int("target_dim", target_dim, target_dim)
            params['hidden_layers'] = trial.suggest_categorical('hidden_layers',  [[2], [4], [8], [2, 2], [4, 4], [8, 8]]) #, [16, 16]])
            params['num_grids'] = trial.suggest_int('num_grids', 4, 12)
            params['lr'] = trial.suggest_loguniform('lr', 1e-4, 1e-2)
            # params['steps'] = trial.suggest_int('steps', 500, 2000)

        model = SurrogateModel(model_name, params)
        
        if cls:
            base_model = model
            model = RegressionToClassificationWrapper(base_model)
            
        model.fit(X_train, y_train)
        preds = model.predict(X_train)

        if cls:
            return accuracy_score(y_train, preds)
        else:
            return mean_squared_error(y_train, preds)

    ### for all kind of Gaussian like model, no need to do hyperparameter_optimization since cost can be high
    if model_name == 'GaussianProcess':
        return {}

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20, n_jobs=-1)
    return study.best_params
