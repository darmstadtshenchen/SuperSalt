import os, ray
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.io import IOManager
from src.surrogate_model import SurrogateModel, hyperparameter_optimization
from src.evaluation import ModelEvaluator
from src.acquisition_function import AcquisitionFunction
import logging

logging.basicConfig(
    filename='mt_record.txt',
    filemode='a',
    format='%(asctime)s\t%(message)s',
    level=logging.INFO
)

# if not ray.is_initialized():
#     ray.init(address='auto')

class MultiTaskBayesianOptimization:
    def __init__(self, data_file, Main_props, correlated_props, feature_props=None, optimization_goal='maximize', scaler_method='standard', 
                 model_list=None, model_path=f'{os.getcwd()}/model_weights', acq_method='ucb',
                 Main_initial_samples=10, correlated_initial_samples=[400],
                 close_pool_threshold=None):
        if Main_props in correlated_props or Main_props == correlated_props:
            raise ValueError('check your Main_props and correlated_props!')
        if len(correlated_props) != len(correlated_initial_samples):
            raise ValueError('check your correlated_initial_samples and correlated_props!')       
        self.data_file = data_file
        self.target_props = Main_props
        self.correlated_props=correlated_props
        self.feature_props = feature_props
        self.optimization_goal = optimization_goal
        self.scaler_method = scaler_method
        self.io_manager = IOManager(method=scaler_method)
        self.model_list = model_list if model_list is not None else ['Ridge', 'Lasso', 'ElasticNet', 'KNeighborsRegressor', 'DecisionTreeRegressor', 'RandomForest', 'SVR', 'MLPRegressor', 'GradientBoostingRegressor', 'AdaBoostRegressor', 'ExtraTreesRegressor', 'XGBoost', 'LightGBM'] # 'LinearRegression' is deprecated
        self.model_path = model_path
        self.correlatedprops_num = len(correlated_props)
        self.Main_initial_samples=Main_initial_samples
        self.acq_method = acq_method

        
        if not ray.is_initialized():
            if os.environ.get('SLURM_JOB_CPUS_PER_NODE', 1) is not None:
                num_cpus = int(os.environ.get('SLURM_JOB_CPUS_PER_NODE', 1))
                memory_per_cpu = int(os.environ.get('SLURM_MEM_PER_CPU', 1)) * 1024 * 1024
                total_memory = num_cpus * memory_per_cpu
                logging.info(f'ray run in slurm: num_cpus: {num_cpus}; memory_per_cpu: {memory_per_cpu}; total_memory: {total_memory}')
                ray.init(num_cpus=num_cpus, _memory=total_memory)
            else:
                ray.init()

        # Data reading and scaling
        _, self.y = self.io_manager.read_data(data_file, target_props=Main_props, feature_props=feature_props)
        self.y = -self.y if self.optimization_goal == 'minimize' else self.y

        # Iterate over each property in correlated_props
        for i, prop in enumerate(correlated_props):
            # Read data for each property
            _, y_correlated_temp = self.io_manager.read_data(data_file, target_props=[prop], feature_props=[])

            # Check optimization goal and adjust y_correlated_temp accordingly
            if self.optimization_goal == 'minimize':
                y_correlated_temp = -y_correlated_temp

            # Dynamically create and assign attribute
            setattr(self, f'y_correlated_{i}', y_correlated_temp)    
            logging.info(f"Attribute y_correlated_{i} set with data (sample)")

        # Read features once if they are the same for all properties
        self.X, _ = self.io_manager.read_data(data_file, target_props= Main_props + correlated_props, feature_props=feature_props)

        self.Main_initial_samples = Main_initial_samples
        

        for i, correlatedIS in enumerate(correlated_initial_samples):
            y_correlated_length = len(getattr(self, f'y_correlated_{i}'))
            correlated_temp_initial_samples = correlatedIS
            # Dynamically create and assign attribute
            setattr(self, f'correlated_{i}_initial_samples', correlated_temp_initial_samples)
            logging.info(f"Attribute correlated_{i}_initial_samples set with data (sample): {correlated_temp_initial_samples}")  

        
        
        if close_pool_threshold is None:
            self.cplb = np.min(self.y, axis=0)
            product = np.prod(self.y-self.cplb, axis=1)
            indexes = np.argsort(product)
            select_index = int(len(self.y)*0.99)
            self.close_pool_threshold = product[indexes][select_index].item()
    

    def close_pooling_test(self, n_bootstrap_sample_nums=20, n_iter=100, batch_size=3, hpar=0.1):

        XMain_train, XMain_candidate, yMain_train, yMain_candidate = train_test_split(
            self.X, self.y, test_size=1 - self.Main_initial_samples / len(self.X)
            )
        for i in range(self.correlatedprops_num):
            initial_samples = getattr(self, f'correlated_{i}_initial_samples')
            y_current = getattr(self, f'y_correlated_{i}')
            # Create the split and dynamically set global variables
            globals()[f'Xcorrelated{i}_train'], globals()[f'Xcorrelated{i}_candidate'], \
            globals()[f'ycorrelated{i}_train'], globals()[f'ycorrelated{i}_candidate'] = train_test_split(
                self.X, y_current, test_size=1 - initial_samples / len(self.X)
            )
        current_best = np.max(np.prod(yMain_train-self.cplb, axis=1))
        while current_best >= self.close_pool_threshold:
            XMain_train, XMain_candidate, yMain_train, yMain_candidate = train_test_split(
                self.X, self.y, test_size=1 - self.Main_initial_samples / len(self.X)
                )
            current_best = np.max(np.prod(yMain_train-self.cplb, axis=1))
        logging.info(current_best)
        logging.info(f'The current_best is {current_best}')
        logging.info(f'Threshold is {self.close_pool_threshold}')
        logging.info(f'Threshold is {self.close_pool_threshold}')
        corr_model_save_paths=[]
        for i in range(self.correlatedprops_num):
            # Retrieve and standardize data
            Xcorrelated_train = globals()[f'Xcorrelated{i}_train']
            ycorrelated_train = globals()[f'ycorrelated{i}_train']
            Xcorrelated_scaled, ycorrelated_scaled = self.io_manager.standardize_data(Xcorrelated_train, ycorrelated_train)
            
            # Update the globals with the new scaled data
            globals()[f'Xcorrelated{i}_scaled'] = Xcorrelated_scaled
            globals()[f'ycorrelated{i}_scaled'] = ycorrelated_scaled
            
            # logging.info the shapes of the scaled data
            logging.info(Xcorrelated_scaled.shape)
            logging.info(ycorrelated_scaled.shape)
            
            # Modify the path and create a ModelEvaluator instance
            modified_path = f'{self.model_path}_correlated{i}'
            corr_model_save_paths.append(modified_path)
            globals()[f'correlated{i}evaluator'] = ModelEvaluator(Xcorrelated_scaled, ycorrelated_scaled, file_path=modified_path)
            
            correlatedevaluator=globals()[f'correlated{i}evaluator']
            globals()[f'correlated{i}modelres']=correlatedevaluator.evaluate_with_stacking(model_names=self.model_list, num_target=0, n_bootstrap_sample_nums=n_bootstrap_sample_nums, cls=False)  #??
            # logging.info(globals()[f'correlated{i}modelres'])
        
    
        for iter in range(n_iter):
            current_best = np.max(np.prod(yMain_train-self.cplb, axis=1))
            current_best_next = 0
            XMain_scaled, yMain_scaled = self.io_manager.standardize_data(XMain_train, yMain_train)

            X_candidate = XMain_candidate.copy()
            candidate_X_scaled = self.io_manager.standardize_data(X_candidate)

            Mainevaluator = ModelEvaluator(XMain_scaled, yMain_scaled, file_path=self.model_path+ '_Main')
            ### stacking model list
            Mainmodel_train = Mainevaluator.MT_train_stacking_model(model_names=self.model_list, corr_model_save_paths=corr_model_save_paths, num_target=0,n_bootstrap_sample_nums=10, cls=False)
            
            acquisition_function = AcquisitionFunction(hpar)
            next_indexes = acquisition_function.MT_select_next(method=self.acq_method, X_candidate=candidate_X_scaled, Mainmodel_train=Mainmodel_train, batch_size=batch_size, y_best=current_best)
    
            logging.info(next_indexes)
            y_next = yMain_candidate[next_indexes]
            current_best_next = np.max(np.prod(y_next-self.cplb, axis=1))
            logging.info(f'train_best and sampling best: {current_best}, {current_best_next}')

            logging.info(f'train_best and sampling best: {current_best}, {current_best_next}')
            logging.info(f'Threshold is {self.close_pool_threshold}')(f'{i}\t{len(yMain_train)}\t{np.mean(y_next-self.cplb)}\t{np.std(y_next-self.cplb)}\t{current_best_next}\t{current_best}\n')

            XMain_train = np.vstack([XMain_train, XMain_candidate[next_indexes]])
            yMain_train = np.vstack([yMain_train, y_next])
            XMain_candidate = np.delete(XMain_candidate, next_indexes, axis=0)
            yMain_candidate = np.delete(yMain_candidate, next_indexes, axis=0)

            current_best = np.max(np.prod(yMain_train-self.cplb, axis=1))

            if current_best >= self.close_pool_threshold:
                logging.info((f"Threshold {self.close_pool_threshold} reached at iteration {iter+1}. The optimum target value is {current_best}"))
                logging.info(f"Threshold {self.close_pool_threshold} reached at iteration {iter+1}. The optimum target value is {current_best}")
                break