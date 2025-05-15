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
    filename='mf_record.txt',
    filemode='a',
    format='%(asctime)s\t%(message)s',
    level=logging.INFO
)
# if not ray.is_initialized():
#     ray.init(address='auto')

class MultiFidelityBayesianOptimization:
    def __init__(self, data_file, HF_props, LF_props, feature_props=None, optimization_goal='maximize', scaler_method='standard', 
                 model_list=None, model_path=f'{os.getcwd()}/model_weights', acq_method='ucb',
                 HF_initial_samples=10, LF_initial_samples=[20],HFcost=10,LFcost=[1],
                 close_pool_threshold=None):
        if HF_props in LF_props or HF_props == LF_props:
            raise ValueError('check your HF_props and LF_props!')
        if len(LF_props) != len(LF_initial_samples):
            raise ValueError('check your LF_initial_samples and LF_props!')
        if len(LF_props) != len(LFcost):
            raise ValueError('check your LF cost setting')        
        self.data_file = data_file
        # self.target_props = HF_props
        self.LF_props=LF_props
        self.feature_props = feature_props
        self.optimization_goal = optimization_goal
        self.scaler_method = scaler_method
        self.io_manager = IOManager(method=scaler_method)
        self.model_list = model_list if model_list is not None else ['Ridge', 'Lasso', 'ElasticNet', 'KNeighborsRegressor', 'DecisionTreeRegressor', 'RandomForest', 'SVR', 'MLPRegressor', 'GradientBoostingRegressor', 'AdaBoostRegressor', 'ExtraTreesRegressor', 'XGBoost', 'LightGBM'] # 'LinearRegression' is deprecated
        self.model_path = model_path
        self.LFprops_num = len(LF_props)
        self.HF_initial_samples=HF_initial_samples
        self.HFcost=HFcost
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
        _, self.y = self.io_manager.read_data(data_file, target_props=HF_props, feature_props=[])
        self.y = -self.y if self.optimization_goal == 'minimize' else self.y

        # Iterate over each property in LF_props
        for i, prop in enumerate(LF_props):
            # Read data for each property
            _, y_LF_temp = self.io_manager.read_data(data_file, target_props=[prop], feature_props=[])

            # Check optimization goal and adjust y_LF_temp accordingly
            if self.optimization_goal == 'minimize':
                y_LF_temp = -y_LF_temp
                

            # Dynamically create and assign attribute
            setattr(self, f'y_LF_{i}', y_LF_temp)    
            logging.info(f"Attribute y_LF_{i} set with data (sample)")

        # Read features once if they are the same for all properties
        self.X, _ = self.io_manager.read_data(data_file, target_props=HF_props + LF_props, feature_props=None)

        self.HF_initial_samples = HF_initial_samples
        self.cost=[]
        self.cost.append(HFcost)
        for i, LFIS in enumerate(LF_initial_samples):
            y_LF_length = len(getattr(self, f'y_LF_{i}'))
            LF_temp_initial_samples = LFIS
            # Dynamically create and assign attribute
            setattr(self, f'LF_{i}_initial_samples', LF_temp_initial_samples)
            logging.info(f"Attribute LF_{i}_initial_samples set with data (sample): {LF_temp_initial_samples}")  
        for i, LFc in enumerate(LFcost):
            LF_temp_cost = LFc
            self.cost.append(LF_temp_cost)
            # Dynamically create and assign attribute
            setattr(self, f'LF_{i}_cost', LF_temp_cost)
            logging.info(f"Attribute LF_{i}_cost: {LF_temp_cost}") 
            logging.info(self.cost) 
        
        
        if close_pool_threshold is None:
            self.cplb = np.min(self.y, axis=0)
            product = np.prod(self.y-self.cplb, axis=1)
            indexes = np.argsort(product)
            select_index = int(len(self.y)*0.99)
            self.close_pool_threshold = product[indexes][select_index].item()

    def close_pooling_test(self, n_bootstrap_sample_nums=20, n_iter=100, batch_size=10, hpar=0.1):

        HFindices = np.arange(len(self.y))
        XHF_train, XHF_candidate, yHF_train, yHF_candidate, HFidx_train, HFidx_candidate = train_test_split(
            self.X, self.y, HFindices, test_size=1 - self.HF_initial_samples / len(self.X)
            )
        for i in range(self.LFprops_num):
            initial_samples = getattr(self, f'LF_{i}_initial_samples')
            y_current = getattr(self, f'y_LF_{i}')
            # Create the split and dynamically set global variables
            globals()[f'XLF{i}_train'], globals()[f'XLF{i}_candidate'], \
            globals()[f'yLF{i}_train'], globals()[f'yLF{i}_candidate'], \
            globals()[f'LF{i}idx_train'],globals()[f'LF{i}idx_candidate']= train_test_split(
                self.X, y_current, HFindices, test_size=1 - initial_samples / len(self.X)
            )
        current_best = np.max(np.prod(yHF_train-self.cplb, axis=1))
        while current_best >= self.close_pool_threshold:
            XHF_train, XHF_candidate, yHF_train, yHF_candidate, HFidx_train, HFidx_candidate = train_test_split(
                self.X, self.y, HFindices, test_size=1 - self.HF_initial_samples / len(self.X)
                )
            current_best = np.max(np.prod(yHF_train-self.cplb, axis=1))
        logging.info(current_best)
        logging.info(f'Current best is {current_best}')
        logging.info(f'Threshold is {self.close_pool_threshold}')
        logging.info(f'Threshold is {self.close_pool_threshold}')
        acquisition_function = AcquisitionFunction(hpar)
        for iter in range(n_iter):
            current_best = np.max(np.prod(yHF_train-self.cplb, axis=1))
            current_best_next = 0
            XHF_scaled, yHF_scaled = self.io_manager.standardize_data(XHF_train, yHF_train, if_train=True)
            for i in range(self.LFprops_num):
                    # Fetch the data from globals using the correct current names
                    XLF_train = globals()[f'XLF{i}_train']
                    yLF_train = globals()[f'yLF{i}_train']
                    # Standardize data
                    XLF_scaled, yLF_scaled = self.io_manager.standardize_data(XLF_train, yLF_train, if_train=True)
                    # Update the globals with the new scaled data
                    globals()[f'XLF{i}_scaled'] = XLF_scaled
                    globals()[f'yLF{i}_scaled'] = yLF_scaled

            X_candidate = XHF_candidate.copy()
            candidate_X_scaled = self.io_manager.standardize_data(X_candidate, if_train=True)

            HFevaluator = ModelEvaluator(XHF_scaled, yHF_scaled, file_path=self.model_path+ '_HF')
            for i in range(self.LFprops_num):
                    XLF_scaled = globals()[f'XLF{i}_scaled']
                    yLF_scaled = globals()[f'yLF{i}_scaled']
                    # logging.info(XLF_scaled.shape)
                    # logging.info(yLF_scaled.shape)
                    modified_path = f'{self.model_path}_LF{i}'
                    globals()[f'LF{i}evaluator']=ModelEvaluator(XLF_scaled, yLF_scaled, file_path=modified_path)
            HFmodelres = HFevaluator.evaluate(model_names=self.model_list, num_target=0,n_bootstrap_sample_nums=n_bootstrap_sample_nums, cls=False)

            for i in range(self.LFprops_num):
                LFevaluator=globals()[f'LF{i}evaluator']
                globals()[f'LF{i}modelres']=LFevaluator.evaluate(model_names=self.model_list, num_target=0,n_bootstrap_sample_nums=n_bootstrap_sample_nums, cls=False)

            acquisition_function = AcquisitionFunction(hpar)
            HFpathtest=self.model_path+ '_HF'
            HFres=acquisition_function.MF_predres(
                                                X_candidates=candidate_X_scaled, 
                                                model_name_list=self.model_list, 
                                                model_path=HFpathtest,
                                                    model_result=None, stack=False
                                                    )
            allLFres = []
            for i in range(self.LFprops_num):
                modified_path = f'{self.model_path}_LF{i}'
                globals()[f'LF{i}res']=acquisition_function.MF_predres(
                                                X_candidates=candidate_X_scaled, 
                                                model_name_list=self.model_list, 
                                                model_path=modified_path,
                                                    model_result=None, stack=False
                                                    )
                allLFres.append(globals()[f'LF{i}res'])

            mean_tuple = (HFres[0],) + tuple(LFres[0] for LFres in allLFres)
            std_tuple = (HFres[1],) + tuple(LFres[1] for LFres in allLFres)

            next_indexes, preferredlevel = acquisition_function.BOfusion_select_next(method=self.acq_method, HFidx_candidate=HFidx_candidate,
                                                                                mean_tuple=mean_tuple, std_tuple=std_tuple,cost=self.cost,
                                                                                batch_size=batch_size, y_best=current_best)

            for i in range(self.LFprops_num):
                LFidx_train = globals()[f'LF{i}idx_train']  
                for j, idx in enumerate(next_indexes):
                    if idx in LFidx_train:
                        preferredlevel[j] = 0
            logging.info(next_indexes)
            logging.info(preferredlevel)
            logging.info(f'next_indexes is {next_indexes}')
            logging.info(f'preferredlevel is {preferredlevel}')
            mask0 = preferredlevel == 0
            next_indexHF = next_indexes[mask0]

            if next_indexHF.size > 0:
                positions = [np.where(HFidx_candidate == idx)[0][0] for idx in next_indexHF if idx in HFidx_candidate]

                XHF_samples = XHF_candidate[positions]
                yHF_samples = yHF_candidate[positions]

                XHF_train = np.vstack([XHF_train, XHF_samples])
                yHF_train = np.vstack([yHF_train, yHF_samples])

                XHF_candidate = np.delete(XHF_candidate, positions, axis=0)
                yHF_candidate = np.delete(yHF_candidate, positions, axis=0)

                HFidx_train = np.concatenate([HFidx_train, HFidx_candidate[positions]])
                HFidx_candidate = np.delete(HFidx_candidate, positions)
                current_best_next = np.max(np.prod(yHF_samples-self.cplb, axis=1))
                logging.info(f'train_best and sampling best: {current_best}, {current_best_next}')
                logging.info(f'train_best and sampling best: {current_best}, {current_best_next}')                
            else:
                logging.info(f'train_best and sampling best: {current_best}, not sampling in HF this iteration')
                logging.info(f'train_best and sampling best: {current_best}, not sampling in HF this iteration')          
            for i in range(self.LFprops_num):
                mask = preferredlevel == (i + 1)
                next_indexLF = next_indexes[mask]

                if next_indexLF.size > 0:
                    positionsLF = [np.where(globals()[f'LF{i}idx_candidate'] == idx)[0][0] for idx in next_indexLF if idx in globals()[f'LF{i}idx_candidate']]

                    XLF_samples = globals()[f'XLF{i}_candidate'][positionsLF]
                    yLF_samples = globals()[f'yLF{i}_candidate'][positionsLF]

                    globals()[f'XLF{i}_train'] = np.vstack([globals()[f'XLF{i}_train'], XLF_samples])
                    globals()[f'yLF{i}_train'] = np.vstack([globals()[f'yLF{i}_train'], yLF_samples])

                    globals()[f'XLF{i}_candidate'] = np.delete(globals()[f'XLF{i}_candidate'], positionsLF, axis=0)
                    globals()[f'yLF{i}_candidate'] = np.delete(globals()[f'yLF{i}_candidate'], positionsLF, axis=0)

                    globals()[f'LF{i}idx_train'] = np.concatenate([globals()[f'LF{i}idx_train'], globals()[f'LF{i}idx_candidate'][positionsLF]])
                    globals()[f'LF{i}idx_candidate'] = np.delete(globals()[f'LF{i}idx_candidate'], positionsLF)


            if current_best_next >= self.close_pool_threshold:
                logging.info(f"Threshold {self.close_pool_threshold} reached at iteration {iter+1}. The optimum target value is {current_best_next}")
                logging.info(f"Threshold {self.close_pool_threshold} reached at iteration {iter+1}. The optimum target value is {current_best_next}")
                break 