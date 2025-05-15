import os, ray, pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.io import IOManager
from src.surrogate_model import SurrogateModel, hyperparameter_optimization
from src.evaluation import ModelEvaluator
from src.acquisition_function import AcquisitionFunction
from src.sampling import Sampler
import logging
from datetime import datetime

import torch
from torch.quasirandom import SobolEngine
from botorch.utils.transforms import normalize, unnormalize

logging.basicConfig(
    filename='bo.log',
    filemode='a',
    format='%(asctime)s\t%(message)s',
    level=logging.INFO
)

class BayesianOptimization:
    def __init__(self, target_props, data_file=None, feature_props=None, optimization_goal='maximize', scaler_method='standard', model_list=None, model_path=f'{os.getcwd()}/model_weights', stacking=True, acq_method='ucb', candidate_file=None, close_pool=False, close_pool_initial_samples=10, close_pool_threshold=None, select_region=None):
        self.data_file = data_file
        self.target_props = target_props
        self.feature_props = feature_props
        self.optimization_goal = optimization_goal
        self.scaler_method = scaler_method
        self.io_manager = IOManager(method=scaler_method)
        self.model_list = model_list if model_list is not None else ['Ridge', 'Lasso', 'ElasticNet', 'KNeighborsRegressor', 'DecisionTreeRegressor', 'RandomForest', 'SVR', 'MLPRegressor', 'GradientBoostingRegressor', 'AdaBoostRegressor', 'ExtraTreesRegressor', 'XGBoost', 'LightGBM', 'GaussianProcess', 'FastKAN'] # 'LinearRegression' is deprecated
        self.model_path = model_path
        self.stacking = stacking
        self.acq_method = acq_method
        self.select_region = select_region

        if not ray.is_initialized():
            logging.info('initial ray')
            if os.environ.get('SLURM_JOB_CPUS_PER_NODE') is not None:
                num_cpus = int(os.environ.get('SLURM_JOB_CPUS_PER_NODE'))
                memory_per_cpu = int(os.environ.get('SLURM_MEM_PER_CPU')) * 1024 * 1024
                total_memory = num_cpus * memory_per_cpu
                logging.info(f'please set your tmp dir path in ray.init when you submit HPC jobs')
                selected_path = '/work/scratch/yz44maby/tmp/'
                timestamp = datetime.now().strftime("%H%M%S")
                ray_temp_dir = os.path.join(selected_path, f'ray_{timestamp}')
                os.makedirs(ray_temp_dir, exist_ok=True)
                ray.init(num_cpus=num_cpus, _memory=total_memory, _temp_dir=ray_temp_dir)
                logging.info(f'ray run in slurm: num_cpus: {num_cpus}; memory_per_cpu: {memory_per_cpu}; total_memory: {total_memory}')
            else:
                ray.init(_temp_dir=f'{os.getcwd()}/tmp')
                logging.info(f'ray run in local')

        # Data reading and scaling
        if data_file is not None:
            self.X, self.y = self.io_manager.read_data(data_file, target_props=target_props, feature_props=feature_props, handle_null=True, drop_non_numeric=True)
            self.y = -self.y if self.optimization_goal == 'minimize' else self.y
        if candidate_file is not None:
            self.X_cand = self.io_manager.read_candidate_data(candidate_file, target_props=target_props, feature_props=feature_props, drop_non_numeric=True)
        else:
            self.X_cand = None

        if close_pool:
            self.close_pool_initial_samples = min(close_pool_initial_samples, (len(self.y)//10+1))
            if close_pool_threshold is None:
                if self.select_region is None:
                    # 1. 记录y每个性质的最大值和最小值，以便归一化
                    self.min_vals = np.min(self.y, axis=0)
                    self.max_vals = np.max(self.y, axis=0)
                    self.ranges = self.max_vals - self.min_vals
    
                    # 2. 对y的不同列进行归一化
                    normalized_y = (self.y - self.min_vals) / self.ranges
    
                else:
                    # Calculate distances for each target property separately
                    select_region_mean = np.mean(self.select_region, axis=0)
                    distances = -np.abs(self.y - select_region_mean)
    
                    self.min_vals = np.min(distances, axis=0)
                    self.max_vals = np.max(distances, axis=0)
                    self.ranges = self.max_vals - self.min_vals
    
                    normalized_y = (distances - self.min_vals) / self.ranges

                # 计算归一化后每行的乘积
                product = np.prod(normalized_y, axis=1)
                indexes = np.argsort(product)
                select_index = max(int(len(product) * 0.99), len(product)-3)
                select_index_init = int(len(product) * 0.5)

                # 计算阈值
                self.close_pool_threshold = product[indexes][select_index]
                self.close_pool_init_index = select_index_init
                self.data_index = indexes
                self.close_pool_init_threshold = product[indexes][select_index_init]
            else:
                self.min_vals = np.min(self.y, axis=0)
                self.max_vals = np.max(self.y, axis=0)
                self.ranges = self.max_vals - self.min_vals
                
                normalized_y = (self.y - self.min_vals) / self.ranges
                product = np.prod(normalized_y, axis=1)
                indexes = np.argsort(product)
                select_index = max(int(len(product) * 0.99), len(product)-3)
                select_index_init = int(len(product) * 0.5)

                # 计算阈值
                self.close_pool_threshold = (close_pool_threshold - self.min_vals) / self.ranges
                self.close_pool_init_index = select_index_init
                self.data_index = indexes
                self.close_pool_init_threshold = product[indexes][select_index_init]

    def compute_normalized_product(self, y_values):
        normalized = (y_values - self.min_vals) / self.ranges
        product = np.prod(normalized, axis=1)
        return product

    def compute_normalized_product_region(self, y_values):
        select_region_mean = np.mean(self.select_region, axis=0)
        distances = -np.abs(y_values - select_region_mean)
        normalized = (distances - self.min_vals) / self.ranges
        product = np.prod(normalized, axis=1)
        return product

    def custom_train_test_split(self, random_state=None):

        if not hasattr(self, 'data_index') or not hasattr(self, 'close_pool_init_index'):
            raise ValueError("Ensure 'self.data_index' and 'self.close_pool_init_index' are set.")

        rng = np.random.default_rng(seed=random_state)

        self.close_pool_initial_samples = min(self.close_pool_initial_samples, self.close_pool_init_index)
        train_indices = rng.choice(self.data_index[:self.close_pool_init_index], size=self.close_pool_initial_samples, replace=False)
        candidate_indices = np.setdiff1d(np.arange(len(self.X)), train_indices)
    
        return self.X[train_indices], self.X[candidate_indices], self.y[train_indices], self.y[candidate_indices]

    
    def close_pooling_test(self, n_bootstrap_sample_nums=20, n_iter=100, batch_size=10, hpar=0.1, save_all_info=True, sampling_method='simulated_annealing', num_candidate=100, n_samples=2000, iterations=30, lb=None, ub=None, all_surrogate_sampling=False):
        
        logging.info(f'Threshold is {self.close_pool_threshold}, init_sampling threshold is {self.close_pool_init_threshold}')
        
        X_train, X_candidate, y_train, y_candidate = self.custom_train_test_split()

        products = self.compute_normalized_product(y_train) if self.select_region is None else self.compute_normalized_product_region(y_train)
        current_best = np.max(products)

        with open('performance_record.txt', 'a') as f:
            f.write('iter_num\tnumber_of_samples\tmean_value_of_this_iteration\tstd_of_this_iteration\tbest_value_of_this_iteration\tcurrent_best_value\n')
    
        acquisition_function = AcquisitionFunction(hpar)

        for i in range(n_iter):

            X_scaled, y_scaled, candidate_X_scaled, candidate_y_scaled = self.io_manager.standardize_data(X_train, y_train, X_candidate, y_candidate, if_train=True, data_id=None)

            select_region = self.io_manager.scaler_y.transform(self.select_region) if self.select_region is not None else self.select_region
            model_evaluator = ModelEvaluator(X_scaled, y_scaled, file_path=self.model_path)
            sampler = Sampler(self.io_manager.scaler_X)
            feature_dim = X_scaled.shape[1]
            
            target_model_res = {}
            for target_idx in range(y_train.shape[1]):
                # *** TBD: add the automatice column cls detection
                if self.stacking:
                    modelres = model_evaluator.evaluate_with_stacking(model_names=self.model_list, num_target=target_idx, n_bootstrap_sample_nums=n_bootstrap_sample_nums, cls=False)
                else:
                    modelres = model_evaluator.evaluate(model_names=self.model_list, num_target=target_idx, n_bootstrap_sample_nums=n_bootstrap_sample_nums, cls=False)  
                target_model_res[target_idx] = modelres

            if len(candidate_X_scaled)>5000000:
                logging.info('Candidate screening')
                candi_X_scaled = sampler.generate_candidate_parallel(method=sampling_method, feature_dim=feature_dim, model_results=target_model_res, model_list=self.model_list, num_candidate=num_candidate, n_samples=n_samples, iterations=iterations, candidate_list=candidate_X_scaled, all_surrogate_sampling=all_surrogate_sampling, parallel=False)
            else:
                candi_X_scaled = candidate_X_scaled
                
            ### by adding the 'model_result' parameter, acquisition_function.select_next function can directly using the model saving in computer memory
            next_indexes = acquisition_function.select_next(method=self.acq_method, X_candidate=candi_X_scaled, model_name_list=self.model_list, num_target=y_train.shape[1], model_path=self.model_path, batch_size=batch_size, y_best=current_best, model_result=target_model_res, stack = self.stacking, select_region=select_region)
            
            ### if value of 'model_result' parameter is not provided, function will try to load the saved model from model_weight saving folder
            # next_indexes = acquisition_function.select_next(method=self.acq_method, X_candidate=candi_X_scaled, model_name_list=self.model_list, num_target=y_train.shape[1], model_path=self.model_path, batch_size=batch_size, y_best=current_best, stack = self.stacking)
            
            y_next = y_candidate[next_indexes]
            products_next = self.compute_normalized_product(y_next) if self.select_region is None else self.compute_normalized_product_region(y_next)
            current_best_next = np.max(products_next)
            
            logging.info(f'train_best and sampling best: {current_best}, {current_best_next}')

            with open('performance_record.txt', 'a') as f:
                f.write(f'{i}\t{len(y_train)}\t{np.mean(products_next)}\t{np.std(products_next)}\t{current_best_next}\t{current_best}\n')
            
            if save_all_info:
                os.mkdir(f'{self.model_path}/{i}')
                os.popen(f'mv {self.model_path}/*.pkl {self.model_path}/{i}/')
                with open(f'{self.model_path}/{i}/data.pickle', 'wb') as f:
                    pickle.dump({'train': [X_train, y_train], 'test': [X_candidate, y_candidate]}, f)
            
            
            X_train = np.vstack([X_train, X_candidate[next_indexes]])
            y_train = np.vstack([y_train, y_next])
            X_candidate = np.delete(X_candidate, next_indexes, axis=0)
            y_candidate = np.delete(y_candidate, next_indexes, axis=0)

            products = self.compute_normalized_product(y_train) if self.select_region is None else self.compute_normalized_product_region(y_train)
            current_best = np.max(products)

            if current_best >= self.close_pool_threshold:
                logging.info(f"Threshold {self.close_pool_threshold} reached at iteration {i+1}. The optimum target value is {current_best}, the last selected set which contain maximum value is {y_next}")
                with open('performance_record.txt', 'a') as f:
                    f.write(f"Threshold {self.close_pool_threshold} reached at iteration {i+1}. The optimum target value is {current_best}")
                break


    def optimize(self, batch_size=20, n_bootstrap_sample_nums=20, sampling_method='genetic_algorithm', num_candidate=100, n_samples=1000, iterations=30, hpar=0.1, lb=None, ub=None, all_surrogate_sampling=False, if_train=True, candidate_sampling=False):

        ## initializing
        logging.info('Initialisation')
        X_train, y_train = self.X, self.y

        current_lb = np.min(self.y, axis=0)
        if self.select_region is None:
            current_best = np.max(np.prod(y_train-current_lb, axis=1))
        else:
            select_region_mean = np.mean(self.select_region, axis=0)
            distances = np.abs(y_train - select_region_mean)
            distances_normalized = distances / np.std(distances, axis=0)
            combined_distance = -np.sum(distances_normalized, axis=1)
            current_best = np.max(combined_distance)
            #current_best = np.max(-np.prod(distances, axis=1))
        #current_best = np.max(np.prod(y_train-current_lb, axis=1))
        
        if self.X_cand is None:
            X_scaled, y_scaled = self.io_manager.standardize_data(X=X_train, y=y_train, feature_range=(0, 1), custom_min=None, custom_max=None, if_train=True, data_id=None)
        else:
            # X_cand = self.X_cand[:,1:]
            X_cand = self.X_cand[:,:]
            X_scaled, y_scaled, cand_X_scaled = self.io_manager.standardize_data(X=X_train, y=y_train, cand_X=X_cand, feature_range=(0, 1), custom_min=None, custom_max=None, if_train=True, data_id=None)

        if self.select_region is not None:
            select_region = self.io_manager.scaler_y.transform(self.select_region)
        else:
            select_region = self.select_region
        
        model_evaluator = ModelEvaluator(X_scaled, y_scaled, file_path=self.model_path)
        feature_dim = X_scaled.shape[1]
        sampler = Sampler(self.io_manager.scaler_X)
        acquisition_function = AcquisitionFunction(hpar)

        ## Model fitting
        if if_train:
            logging.info('Fitting')
            target_model_res = {}
            for target_idx in range(y_train.shape[1]):
                # *** TBD: add the automatice column cls detection
                if self.stacking:
                    modelres = model_evaluator.evaluate_with_stacking(model_names=self.model_list, num_target=target_idx, n_bootstrap_sample_nums=n_bootstrap_sample_nums, cls=False)
                else:
                    modelres = model_evaluator.evaluate(model_names=self.model_list, num_target=target_idx, n_bootstrap_sample_nums=n_bootstrap_sample_nums, cls=False)  
                target_model_res[target_idx] = modelres
        else:
            target_model_res = None

        ## Candidate generation
        if self.X_cand is None:
            logging.info('Candidate generation')
            candidate_X_scaled = sampler.generate_candidate_parallel(method=sampling_method, feature_dim=feature_dim, model_results=target_model_res, model_list=self.model_list, num_candidate=num_candidate, n_samples=n_samples, iterations=iterations, candidate_list=None, all_surrogate_sampling=all_surrogate_sampling, parallel=False)
        elif candidate_sampling:
            logging.info('Candidate screening')
            candidate_X_scaled = sampler.generate_candidate_parallel(method=sampling_method, feature_dim=feature_dim, model_results=target_model_res, model_list=self.model_list, num_candidate=num_candidate, n_samples=n_samples, iterations=iterations, candidate_list=cand_X_scaled, all_surrogate_sampling=all_surrogate_sampling, parallel=False)
        else:
            candidate_X_scaled = cand_X_scaled

        ## Candidate selection 
        logging.info('Candidate selection')
        next_indexes = acquisition_function.select_next(method=self.acq_method, X_candidate=candidate_X_scaled, model_name_list=self.model_list, num_target=y_train.shape[1], model_path=self.model_path, batch_size=batch_size, model_result=target_model_res, stack = self.stacking, y_best=current_best, select_region=select_region)
        samples_next = self.io_manager.inverse_transform_X(candidate_X_scaled[next_indexes])
        pd_samples_next = pd.DataFrame(samples_next)
        pd_samples_next.to_csv(f'{os.getcwd()}/suggested_samples.csv', index=False)
        pd_samples_next_indexs = pd.DataFrame(next_indexes)
        pd_samples_next_indexs.to_csv(f'{os.getcwd()}/suggested_samples_indexes.csv', index=False)
        if self.X_cand is not None:
            pd_samples_next_origin = pd.DataFrame(self.X_cand[next_indexes])
            pd_samples_next_origin.to_csv(f'{os.getcwd()}/suggested_samples_original.csv', index=False)
        
        return samples_next, next_indexes



