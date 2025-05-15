import ray, os, logging
import numpy as np
import os, pickle, torch
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import RidgeCV, LogisticRegression, RidgeClassifier, LinearRegression
from sklearn.ensemble import StackingRegressor, StackingClassifier, RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, clone
from src.surrogate_model import SurrogateModel, hyperparameter_optimization

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  ## CUDA not applicatble yet
device = torch.device('cpu')

class AbstractSurrogateModel(BaseEstimator, RegressorMixin):
    def __init__(self, model_name, models):
        self.model_name = model_name
        self.models = models

    @ray.remote
    def model_predict(self, X, model):
        if self.model_name == 'GaussianProcess':
            Gmodel_res = model.model.posterior(torch.tensor(X, dtype=torch.float32))
            Gmean = Gmodel_res.mean.detach().cpu().numpy().reshape(-1)
            # Gstd = torch.sqrt(Gmodel_res.variance).detach().cpu().numpy().reshape(-1)
            return Gmean
        elif self.model_name == 'KAN':
            return model.model(torch.tensor(X, dtype=torch.float32, device=device)).detach().cpu().numpy().flatten()
        elif self.model_name == 'FastKAN':
            return model.model(torch.tensor(X, dtype=torch.float32, device=device)).detach().cpu().numpy().flatten()
        else:
            preds = model.predict(X)
            return preds

    def fit(self, X, y):
        pass  # 已经拟合好，不需要再 fit

    def predict(self, X):
        predictions = ray.get([self.model_predict.remote(self, X, model) for model in self.models])
        predictions = np.array(predictions)
        return np.mean(predictions, axis=0)

    def predict_proba(self, X):
        if hasattr(self.models[0], "predict_proba"):
            probas = np.array([model.predict_proba(X) for model in self.models])
            return np.mean(probas, axis=0)
        else:
            predictions = ray.get([self.model_predict.remote(self, X, model) for model in self.models])
            predictions = np.array(predictions)
            mean_pred = np.mean(predictions, axis=0)
            std_pred = np.std(predictions, axis=0)
            # 使用正态分布模拟概率分布
            lower_bound = mean_pred - 1.96 * std_pred
            upper_bound = mean_pred + 1.96 * std_pred
            probas = np.clip((X - lower_bound) / (upper_bound - lower_bound), 0, 1)
            return np.vstack((1 - probas, probas)).T


class ModelEvaluator:
    def __init__(self, X_train, y_train, file_path=None):
        self.X_train = X_train
        self.y_train = y_train
        self.file_path = file_path if file_path is not None else f'{os.getcwd()}/model_weights'

    def save_models(self, model_name, optimized_params, models, model_errors, file_name, stacking_model=None, stacking_error=None):
        if not os.path.exists(f'{self.file_path}'):
            os.mkdir(f'{self.file_path}')
        with open(f'{self.file_path}/{file_name}', 'wb') as f:
            pickle.dump({'model_name': model_name, 'optimized_params': optimized_params, 'models': models, 'errors': model_errors}, f)

    def load_models(self, file_name):
        with open(f'{self.file_path}/{file_name}', 'rb') as f:
            data = pickle.load(f)
        return data

    def bootstrap_evaluation(self, model_name, optimized_params, num_target, n_bootstrap_sample_nums=20, cls=False, use_full_eval=False, cross_val=False, cv_n_splits=5):
        n_samples = len(self.X_train)
        errors = []
        models = []
        X_bs = self.X_train
        y_bs = self.y_train[:, num_target]
        if n_bootstrap_sample_nums < 2:
            n_bootstrap_sample_nums = 2
        cv_n_splits = min(n_bootstrap_sample_nums, cv_n_splits)

        if cross_val:
            kf = KFold(n_splits=cv_n_splits)
            for train_idx, val_idx in kf.split(X_bs):
                X_train_cv, X_val_cv = X_bs[train_idx], X_bs[val_idx]
                y_train_cv, y_val_cv = y_bs[train_idx], y_bs[val_idx]

                model = SurrogateModel(model_name, optimized_params)
                model.fit(X_train_cv, y_train_cv)
                preds = model.predict(X_val_cv)

                if cls:
                    error = accuracy_score(y_val_cv, preds)
                else:
                    r2_error = np.clip(r2_score(y_val_cv, preds), 0, 1)
                    error = r2_error

                errors.append(error)
                models.append(model)
                
        else:
            if model_name == 'GaussianProcess':
                X_tr, X_te, y_tr, y_te = train_test_split(X_bs, y_bs, test_size=0.8)
                model = SurrogateModel(model_name, optimized_params)
                model.fit(X_tr, y_tr)
                preds = model.predict(X_te)
                
                if cls:
                    error = accuracy_score(y_te, preds)
                else:
                    r2_error = np.clip(r2_score(y_te, preds), 0, 1)
                    error = r2_error

                errors.append(error)
                models.append(model)
                
            else:
                bootstrap_tasks = []
                for i in range(n_bootstrap_sample_nums):
                    bootstrap_indices = np.random.choice(np.arange(n_samples), size=n_samples, replace=True)
                    X_sample = X_bs[bootstrap_indices]
                    y_sample = y_bs[bootstrap_indices]
                    bootstrap_tasks.append(self._train_model.remote(self, model_name, optimized_params, X_sample, y_sample, X_bs, y_bs, bootstrap_indices, cls, use_full_eval))

                results = ray.get(bootstrap_tasks)
                for res in results:
                    models.append(res['model'])
                    errors.append(res['error'])

        # 保存每个模型名称的所有 bootstrap 结果
        self.save_models(model_name, optimized_params, models, errors, f"{model_name}_{num_target}_bootstrap.pkl")

        return [models, errors]

    @ray.remote
    def _train_model(self, model_name, optimized_params, X_sample, y_sample, X_bs, y_bs, bootstrap_indices, cls, use_full_eval):
        model = SurrogateModel(model_name, optimized_params)
        model.fit(X_sample, y_sample)

        if use_full_eval:
            X_eval = X_bs
            y_eval = y_bs
        else:
            eval_indices = np.setdiff1d(np.arange(len(X_bs)), bootstrap_indices)
            X_eval = X_bs[eval_indices] if len(eval_indices) != 0 else X_bs
            y_eval = y_bs[eval_indices] if len(eval_indices) != 0 else y_bs

        preds = model.predict(X_eval)

        if cls:
            error = accuracy_score(y_eval, preds)
        else:
            r2_error = np.clip(r2_score(y_eval, preds), 0, 1)
            error = r2_error

        return {'model': model, 'error': error}

    def evaluate(self, model_names, num_target, n_bootstrap_sample_nums, cls=False, use_full_eval=False, cross_val=False):
        model_results = {}

        for model_name in model_names:
            
            if model_name == 'GaussianProcess' and len(self.X_train) <= 500:
                cross_val = True
            elif model_name == 'GaussianProcess' and len(self.X_train) > 500:
                cross_val = False
            
            # 优化超参数
            optimized_params = hyperparameter_optimization(model_name, self.X_train, self.y_train[:, num_target], cls=cls)
            # 评估模型
            models, errors = self.bootstrap_evaluation(model_name, optimized_params, num_target, n_bootstrap_sample_nums=n_bootstrap_sample_nums, cls=cls, use_full_eval=use_full_eval, cross_val=cross_val)
            model_results[model_name] = {'models': models, 'errors': errors}

        return model_results

    ### possible meta classifiers: RidgeCV, LogisticRegression, RidgeClassifier, LinearRegression, RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
    def train_stacking_model(self, model_results=None, num_target=0, cls=False, meta_classifier=None, use_probas=False, model_name_list=None):
        if model_results is None:
            if model_name_list is None:
                raise ValueError("When model_results is None, model_name_list must be provided.")
            model_results = {}
            for model_name in model_name_list:
                file_path = f"{self.file_path}/{model_name}_{num_target}_bootstrap.pkl"
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                model_results[model_name] = data
        
        base_models = [(model_name, AbstractSurrogateModel(model_name, model_info['models'])) for model_name, model_info in model_results.items()]
        
        if meta_classifier is None:
            meta_classifier = RandomForestClassifier() if cls else RandomForestRegressor()

        if use_probas and cls:
            stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_classifier, stack_method='predict_proba')
        else:
            stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_classifier) if cls else StackingRegressor(estimators=base_models, final_estimator=meta_classifier)

        X_meta = self.X_train
        y_meta = self.y_train[:, num_target]
        stacking_model.fit(X_meta, y_meta)
        
        # 获取基础模型的权重或系数
        if hasattr(stacking_model.final_estimator_, 'coef_'):
            base_model_contributions = stacking_model.final_estimator_.coef_
        elif hasattr(stacking_model.final_estimator_, 'feature_importances_'):
            base_model_contributions = stacking_model.final_estimator_.feature_importances_
        else:
            base_model_contributions = None

        base_model_errors = {}
        for i, (model_name, _) in enumerate(base_models):
            if base_model_contributions is not None:
                contribution_score = base_model_contributions[i]
            else:
                contribution_score = None  # 如果没有权重或特征重要性，就设置为None
            base_model_errors[model_name] = contribution_score
        
        return stacking_model, base_model_errors

    def evaluate_with_stacking(self, model_names, num_target, n_bootstrap_sample_nums, cls=False, use_full_eval=False, cross_val=False, meta_classifier=None, use_probas=False):
        model_results = self.evaluate(model_names, num_target, n_bootstrap_sample_nums, cls=cls, use_full_eval=use_full_eval, cross_val=cross_val)
        stacking_model, base_model_errors = self.train_stacking_model(model_results, num_target, cls=cls, meta_classifier=meta_classifier, use_probas=use_probas, model_name_list=model_names)
        model_results['stacking_error'] = base_model_errors
        model_results['stacking_model'] = stacking_model
        with open(f'{self.file_path}/stacking_results_{num_target}.pkl', 'wb') as f:
            pickle.dump(model_results, f)
        
        return model_results

    def MT_train_stacking_model(self, model_names, corr_model_save_paths, n_bootstrap_sample_nums=20, num_target=0, cls=False, meta_classifier=None, use_probas=False):
        n_samples = len(self.X_train)
        model_results = {}
        for model_name in model_names:
            for path in corr_model_save_paths:
                file_path = f"{path}/{model_name}_{num_target}_bootstrap.pkl"
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                model_results[f'{path[-1]}_{model_name}'] = data

        # print(f'MT_mr')
        # print(model_results.items())
        base_models = [(model_name, AbstractSurrogateModel(model_name, model_info['models'])) for model_name, model_info in model_results.items()]
        # print(base_models)

        X_meta = self.X_train
        y_meta = self.y_train[:, num_target]

        model_tasks = []
        for i in range(n_bootstrap_sample_nums):
        
            bootstrap_indices = np.random.choice(np.arange(n_samples), size=n_samples, replace=True)
            X_sample = X_meta[bootstrap_indices]
            y_sample = y_meta[bootstrap_indices]
            
            if meta_classifier is None:
                meta_classifier = RandomForestClassifier() if cls else RandomForestRegressor()
        
            if use_probas and cls:
                stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_classifier, stack_method='predict_proba')
            else:
                stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_classifier) if cls else StackingRegressor(estimators=base_models, final_estimator=meta_classifier)
                
            stacking_model.fit(X_sample, y_sample)
            model_tasks.append(stacking_model)
            
        if not os.path.exists(f'{self.file_path}'):
            os.mkdir(f'{self.file_path}')
            
        with open(f'{self.file_path}/correlated_stacking_results_{num_target}_bootstrap.pkl', 'wb') as f:
            pickle.dump(model_tasks, f)
    
        return model_tasks