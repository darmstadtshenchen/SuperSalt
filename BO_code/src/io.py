import os, pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.exceptions import NotFittedError
import logging

class IOManager:
    def __init__(self, root=None, method='standard', file_path=None):
        self.root = root if root else os.getcwd()
        self.method = method
        if method == 'standard':
            self.scaler_X = StandardScaler()           
        elif self.method == 'minmax':
            self.scaler_X = MinMaxScaler()
        else:
            raise ValueError("Invalid method. Use 'standard' or 'minmax'.")
        self.scaler_y = StandardScaler()
        self.file_path = file_path if file_path is not None else f'{os.getcwd()}/model_weights'

    def save_scalers(self, data_id):     
        os.makedirs(self.file_path, exist_ok=True)
        scaler_file = os.path.join(self.file_path, f"scaler{data_id}.pkl")
        with open(scaler_file, "wb") as f:
            pickle.dump({"scaler_X": self.scaler_X, "scaler_y": self.scaler_y}, f)
        logging.info(f"scalers{data_id} saved to {self.file_path} directory.")

    def load_scalers(self, data_id):
        try:
            scaler_file = os.path.join(self.file_path, f"scaler{data_id}.pkl")
            with open(scaler_file, "rb") as f:
                scalers = pickle.load(f)
            self.scaler_X = scalers["scaler_X"]
            self.scaler_y = scalers["scaler_y"]
            logging.info(f"scalers{data_id} loaded from {self.file_path} directory.")
        except FileNotFoundError:
            raise FileNotFoundError(f"scalers{data_id} files not found in {self.file_path} directory or {self.file_path} directory not exists.")

    def read_data(self, file_name, target_props, feature_props=None, drop_columns=None, descriptor_type='magpie', handle_null=True, drop_non_numeric=True):
        
        file_path = os.path.join(self.root, f'{file_name}')
        data = pd.read_csv(file_path)
        if drop_columns is not None:
            logging.info(f"drop columns: {drop_columns}")
            data = data.drop(columns=drop_columns)

        # 检查指定的列是否存在
        missing_cols = [col for col in target_props if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

        target_props = list(set(target_props))  # Remove duplicates if any

        # Check for null values
        if data.isnull().values.any():
            logging.info(f'data contains null value!')
            if not handle_null:
                raise ValueError("Data contains null values. Please handle them or set handle_null to True.")
            else:
                data, non_numeric_columns = self.handle_null_values(data, target_props, drop_non_numeric=drop_non_numeric)
        else:
            data, non_numeric_columns = self.handle_null_values(data, target_props, drop_non_numeric=drop_non_numeric)

        # 如果没有指明feature_props, 使用除target外所有列
        if feature_props is None:
            feature_props = [col for col in data.columns if col not in target_props]
            feature_props = [col for col in feature_props if col not in non_numeric_columns]
        logging.info(f'used feature set: {feature_props}')
        
        # 检查指定的feature列是否存在
        missing_feature_cols = [col for col in feature_props if col not in data.columns]
        if missing_feature_cols:
            raise ValueError(f"Missing feature columns: {missing_feature_cols}")

        X = data[feature_props].to_numpy()
        y = data[target_props].to_numpy()

        if y.ndim == 1:
            y = np.expand_dims(y, -1)

        return X, y

    def read_candidate_data(self, file_name, target_props, feature_props=None, descriptor_type='magpie', drop_non_numeric=True):
        file_path = os.path.join(self.root, f'{file_name}')
        data = pd.read_csv(file_path)

        # 检查指定的列是否存在
        missing_cols = [col for col in target_props if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

        target_props = list(set(target_props))  # Remove duplicates if any

        data, non_numeric_columns = self.handle_null_values(data, target_props, drop_non_numeric=drop_non_numeric, if_train_data=False)

        # 如果没有指明feature_props, 使用除target外所有列
        if feature_props is None:
            feature_props = [col for col in data.columns if col not in target_props]
            feature_props = [col for col in feature_props if col not in non_numeric_columns]
        logging.info(f'used feature set: {feature_props}')
        
        # 检查指定的feature列是否存在
        missing_feature_cols = [col for col in feature_props if col not in data.columns]
        if missing_feature_cols:
            raise ValueError(f"Missing feature columns: {missing_feature_cols}")

        X = data[feature_props].to_numpy()

        return X

    def handle_null_values(self, data, target_props, drop_non_numeric, if_train_data=True):
        # 处理target列的null值，通过删除包含null的行    
        if if_train_data:
            for target in target_props:
                if data[target].isnull().any():
                    data = data.dropna(subset=[target])   # 删除含有空值的行
                    logging.info(f'drop samples contains null properties: {target}')
                if drop_non_numeric and not pd.api.types.is_numeric_dtype(data[target]): 
                    unique_values = data[target].nunique() # 计算唯一值数量
                    if unique_values == 2:
                        data[target] = data[target].astype('category').cat.codes # 将分类数据转换为 0/1 编码
                    else:
                        # 如果不是二分类数据，抛出错误
                        raise ValueError(f"Target column {target} contains non-numeric data that cannot be converted to binary classification.")
        non_numeric_columns = []
        # 处理非target列的null值，通过删除包含null的行
        for column in data.columns:
            if column not in target_props: # 对于非目标列
                if drop_non_numeric and not pd.api.types.is_numeric_dtype(data[column]):
                    logging.info(f'drop feature which is non numeric: {column}')
                    data = data.drop(columns=[column]) # 直接删除非数值列
                    # 记录非数值列列名
                    non_numeric_columns.append(column)  
                elif data[column].isnull().any(): # 处理空值
                    # logging.info(f'drop feature column contains null: {column}')
                    # data = data.drop(columns=[column]) # 直接删除非数值列
                    logging.info(f'drop samples contains null features: {column}')
                    data = data.dropna(subset=[column]) # 删除含有空值的行
        logging.info(f'length of cleaned data: {len(data)}')

        return data, non_numeric_columns


    def standardize_data(self, X=None, y=None, cand_X=None, cand_y=None, feature_range=(0, 1), custom_min=None, custom_max=None, if_train=False, data_id=None):
        """
        Standardize or scale data based on the chosen method (standard/minmax).
        Args:
            X: Training features (optional).
            y: Training targets (optional).
            cand_X: Candidate features for prediction (optional).
            cand_y: Candidate targets for prediction (optional).
            feature_range: Feature range for MinMaxScaler.
            custom_min, custom_max: Custom scaling ranges for MinMaxScaler.
            if_train: Flag to indicate whether it's training mode (default=False).
        Returns:
            Tuple of scaled inputs in the same order as provided.
        """
        assert not (X is None and y is None and cand_X is None and cand_y is None), \
            "At least one of X, y, cand_X, or cand_y must be provided."
    
        # Initialize variables
        X_scaled, y_scaled, cand_X_scaled, cand_y_scaled = None, None, None, None
    
        if if_train:
            # Training mode: fit scalers and save them
            if self.method == 'standard':
                if X is not None:
                    self.scaler_X.fit(X)
                    X_scaled = self.scaler_X.transform(X)
                if y is not None:
                    self.scaler_y.fit(y)
                    y_scaled = self.scaler_y.transform(y)
            elif self.method == 'minmax':
                self.scaler_X.feature_range = feature_range
                if X is not None:
                    self.scaler_X.fit(X)
                    X_scaled = self.scaler_X.transform(X)
                if y is not None:
                    self.scaler_y.fit(y)
                    y_scaled = self.scaler_y.transform(y)
            else:
                raise ValueError("Invalid method. Use 'standard' or 'minmax'.")

            if cand_X is not None:
                cand_X_scaled = self.scaler_X.transform(cand_X)
            if cand_y is not None:
                cand_y_scaled = self.scaler_y.transform(cand_y)
    
            # Save scalers after fitting
            self.save_scalers(data_id)
        else:
            # Prediction mode: load scalers and transform data
            self.load_scalers(data_id)
    
            if X is not None:
                X_scaled = self.scaler_X.transform(X)
            if y is not None:
                y_scaled = self.scaler_y.transform(y)
            if cand_X is not None:
                cand_X_scaled = self.scaler_X.transform(cand_X)
            if cand_y is not None:
                cand_y_scaled = self.scaler_y.transform(cand_y)
    
        # Dynamically construct the return tuple based on input arguments
        return_tuple = tuple(var for var in [X_scaled, y_scaled, cand_X_scaled, cand_y_scaled] if var is not None)
    
        # If there's only one element, return it directly
        return return_tuple[0] if len(return_tuple) == 1 else return_tuple


    def inverse_transform_X(self, X_scaled):
        return self.scaler_X.inverse_transform(X_scaled)

    def inverse_transform_y(self, y_scaled):
        return self.scaler_y.inverse_transform(y_scaled)

    def save_predictions(self, predictions, file_name):
        df = pd.DataFrame(predictions, columns=['Predictions'])
        df.to_csv(file_name, index=False)


