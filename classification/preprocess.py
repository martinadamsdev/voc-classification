"""
VOC气体分类 - 数据预处理模块

提供数据加载、清洗、特征提取和数据分割功能，使用纯函数式风格
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import os
from logger.logger import Logger

# 创建日志记录器
logger = Logger('Preprocess')

def load_data(file_path):
    """
    加载VOC气体数据集
    
    参数:
        file_path: 数据集文件路径
    
    返回:
        加载的数据集DataFrame
    """
    try:
        # 加载数据，没有表头
        data = pd.read_csv(file_path, header=None)
        
        # 添加列名
        # 第一列为气体名称，第二列为浓度
        column_names = ['gas', 'concentration']
        
        # 添加传感器响应列名
        sensors = ['TGS2603', 'TGS2630', 'TGS813', 'TGS822', 'MQ-135', 'MQ-137', 'MQ-138', '2M012', 'VOCS-P', '2SH12']
        for sensor in sensors:
            for i in range(900):  # 每个传感器900个采样点
                column_names.append(f"{sensor}_{i}")
        
        data.columns = column_names
        
        logger.info(f"成功加载数据集，共{data.shape[0]}条记录，{data.shape[1]}个特征")
        return data
    except Exception as e:
        logger.error(f"加载数据集失败: {e}")
        return None

def explore_data(data):
    """
    探索性数据分析
    
    参数:
        data: 数据集DataFrame
    
    返回:
        数据集的基本统计信息
    """
    # 基本信息
    logger.info("数据集基本信息:")
    info_str = str(data.info())
    logger.info(info_str)
    
    # 统计描述
    logger.info("数据集统计描述:")
    desc_str = str(data.describe())
    logger.info(desc_str)
    
    # 检查缺失值
    missing_values = data.isnull().sum()
    logger.info("缺失值统计:")
    missing_str = str(missing_values[missing_values > 0])
    logger.info(missing_str)
    
    # 气体类型分布
    gas_distribution = data['gas'].value_counts()
    logger.info(f"气体类型分布:\n{gas_distribution}")
    
    # 浓度分布
    concentration_distribution = data['concentration'].value_counts()
    logger.info(f"浓度分布:\n{concentration_distribution}")
    
    return data.describe()

def extract_features(data):
    """
    从传感器响应中提取特征
    
    参数:
        data: 数据集DataFrame
    
    返回:
        包含提取特征的DataFrame
    """
    logger.info("开始从传感器响应中提取特征")
    
    # 提取气体类型和浓度
    features = {
        'gas': data['gas'].values,  # 转换为一维数组
        'concentration': data['concentration'].values,  # 转换为一维数组
        'concentration_value': data['concentration'].str.extract('(\d+)').astype(int).values.flatten()  # 确保是一维数组
    }
    
    # 传感器列表
    sensors = ['TGS2603', 'TGS2630', 'TGS813', 'TGS822', 'MQ-135', 'MQ-137', 'MQ-138', '2M012', 'VOCS-P', '2SH12']
    
    # 为每个传感器提取统计特征
    for sensor in sensors:
        # 获取当前传感器的所有响应列
        sensor_columns = [col for col in data.columns if col.startswith(f"{sensor}_")]
        sensor_data = data[sensor_columns]
        
        # 提取统计特征 - 确保所有特征都是一维数组
        features[f"{sensor}_mean"] = sensor_data.mean(axis=1).values
        features[f"{sensor}_std"] = sensor_data.std(axis=1).values
        features[f"{sensor}_max"] = sensor_data.max(axis=1).values
        features[f"{sensor}_min"] = sensor_data.min(axis=1).values
        features[f"{sensor}_range"] = (sensor_data.max(axis=1) - sensor_data.min(axis=1)).values
        features[f"{sensor}_median"] = sensor_data.median(axis=1).values
        features[f"{sensor}_q25"] = sensor_data.quantile(0.25, axis=1).values
        features[f"{sensor}_q75"] = sensor_data.quantile(0.75, axis=1).values
        features[f"{sensor}_iqr"] = (sensor_data.quantile(0.75, axis=1) - sensor_data.quantile(0.25, axis=1)).values
        
        # 提取时域特征
        # 计算上升时间（从10%到90%的响应时间）
        rise_times = []
        for _, row in sensor_data.iterrows():
            values = row.values
            min_val = np.min(values)
            max_val = np.max(values)
            threshold_low = min_val + 0.1 * (max_val - min_val)
            threshold_high = min_val + 0.9 * (max_val - min_val)
            
            # 找到第一个超过低阈值的索引
            try:
                idx_low = np.where(values > threshold_low)[0][0]
            except IndexError:
                idx_low = 0
                
            # 找到第一个超过高阈值的索引
            try:
                idx_high = np.where(values > threshold_high)[0][0]
            except IndexError:
                idx_high = len(values) - 1
                
            rise_time = idx_high - idx_low
            rise_times.append(rise_time)
        
        features[f"{sensor}_rise_time"] = np.array(rise_times)  # 确保是一维数组
        
        # 计算稳态响应（最后100个点的平均值）
        features[f"{sensor}_steady_state"] = sensor_data.iloc[:, -100:].mean(axis=1).values
    
    # 一次性创建DataFrame，避免碎片化
    features_df = pd.DataFrame(features)
    
    logger.info(f"特征提取完成，共提取{features_df.shape[1]-2}个特征")
    
    return features_df

def preprocess_data(data, target_column='gas', test_size=0.5, random_state=42, scaling_method='standard'):
    """
    预处理数据
    
    参数:
        data: 输入数据
        target_column: 目标变量列名
        test_size: 测试集比例
        random_state: 随机种子
        scaling_method: 缩放方法，可选 'standard', 'minmax', 'robust', 'none'
    
    返回:
        X_train, X_test, y_train, y_test, scaler, feature_names, class_names
    """
    # 记录传入的参数
    print(f"preprocess_data 函数接收到的参数: test_size={test_size}")
    
    # 提取特征
    features_df = extract_features(data)
    
    # 对气体类型进行编码
    label_encoder = LabelEncoder()
    features_df['gas_encoded'] = label_encoder.fit_transform(features_df[target_column])
    class_names = label_encoder.classes_
    
    # 分离特征和目标变量
    X = features_df.drop([target_column, 'concentration', 'gas_encoded'], axis=1)
    y = features_df['gas_encoded']
    
    # 保存特征名称
    feature_names = X.columns.tolist()
    
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # 记录分割后的数据集大小
    print(f"数据集分割完成: 训练集 {X_train.shape[0]} 样本 ({(1-test_size)*100:.0f}%), 测试集 {X_test.shape[0]} 样本 ({test_size*100:.0f}%)")
    
    # 特征缩放
    if scaling_method == 'standard':
        scaler = StandardScaler()
    elif scaling_method == 'minmax':
        scaler = MinMaxScaler()
    elif scaling_method == 'robust':
        scaler = RobustScaler()
    else:
        scaler = None
    
    if scaler:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    print(f"类别: {', '.join(class_names)}")
    
    return X_train, X_test, y_train, y_test, scaler, feature_names, class_names

def select_features(X_train, y_train, method='selectkbest', k=10, feature_names=None):
    """
    特征选择
    
    参数:
        X_train: 训练集特征
        y_train: 训练集标签
        method: 特征选择方法，'selectkbest', 'rfe', 或 'rf_importance'
        k: 选择的特征数量
        feature_names: 特征名称列表
    
    返回:
        选择的特征索引和名称
    """
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
    
    if method == 'selectkbest':
        selector = SelectKBest(f_classif, k=k)
        selector.fit(X_train, y_train)
        selected_indices = selector.get_support(indices=True)
        
    elif method == 'rfe':
        estimator = RandomForestClassifier(random_state=42)
        selector = RFE(estimator, n_features_to_select=k)
        selector.fit(X_train, y_train)
        selected_indices = selector.get_support(indices=True)
        
    elif method == 'rf_importance':
        # 使用随机森林特征重要性
        rf = RandomForestClassifier(random_state=42)
        rf.fit(X_train, y_train)
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1][:k]  # 选择前k个重要特征
        selected_indices = indices
    else:
        # 如果没有指定有效的方法，则选择所有特征
        selected_indices = np.arange(X_train.shape[1])
    
    # 获取选择的特征名称
    selected_feature_names = [feature_names[i] for i in selected_indices]
    
    logger.info(f"特征选择完成，从{X_train.shape[1]}个特征中选择了{len(selected_indices)}个特征")
    logger.info(f"选择的特征: {', '.join(selected_feature_names[:10])}...")
    
    return selected_indices, selected_feature_names

def visualize_feature_importance(X_train, y_train, feature_names, n_features=20, save_path='figures/feature_importance.png'):
    """
    可视化特征重要性
    
    参数:
        X_train: 训练集特征
        y_train: 训练集标签
        feature_names: 特征名称列表
        n_features: 显示的特征数量
        save_path: 保存路径
    """
    # 使用随机森林计算特征重要性
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    
    # 获取特征重要性
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # 绘制特征重要性图 - 不设置中文字体，只使用Times New Roman
    plt.figure(figsize=(12, 8))
    plt.title("Feature Importance")
    plt.barh(range(min(n_features, len(indices))), 
             importances[indices[:n_features]], 
             align='center')
    plt.yticks(range(min(n_features, len(indices))), 
               [feature_names[i] for i in indices[:n_features]])
    plt.gca().invert_yaxis()  # 从上到下显示重要性递减
    plt.xlabel('Importance')
    plt.tight_layout()
    
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path)
    plt.close()
    
    logger.info(f"特征重要性可视化完成，显示了前{n_features}个重要特征")
    logger.info(f"特征重要性图已保存到: {save_path}")
    
    return indices, importances