import numpy as np
import pickle
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import json
import logging
from keras.api.models import Sequential, Model
from keras.api.layers import Dense, Dropout, Input
from keras.api.optimizers import Adam
from keras.api.utils import to_categorical

logger = logging.getLogger(__name__)

def train_knn(X_train, y_train, n_neighbors=5, weights='uniform', metric='minkowski'):
    """
    训练K近邻分类器
    
    参数:
        X_train: 训练集特征
        y_train: 训练集标签
        n_neighbors: 邻居数量
        weights: 权重类型，'uniform'或'distance'
        metric: 距离度量方式
    
    返回:
        训练好的KNN模型
    """
    start_time = time.time()
    
    knn = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        metric=metric
    )
    
    knn.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f"KNN模型训练完成，耗时: {training_time:.2f}秒")
    
    return knn, training_time

def train_svm(X_train, y_train, C=1.0, kernel='rbf', gamma='scale'):
    """
    训练支持向量机分类器
    
    参数:
        X_train: 训练集特征
        y_train: 训练集标签
        C: 正则化参数
        kernel: 核函数类型
        gamma: 核系数
    
    返回:
        训练好的SVM模型
    """
    start_time = time.time()
    
    svm = SVC(
        C=C,
        kernel=kernel,
        gamma=gamma,
        probability=True
    )
    
    svm.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f"SVM模型训练完成，耗时: {training_time:.2f}秒")
    
    return svm, training_time

def train_random_forest(X_train, y_train, n_estimators=100, max_depth=None, min_samples_split=2):
    """
    训练随机森林分类器
    
    参数:
        X_train: 训练集特征
        y_train: 训练集标签
        n_estimators: 决策树数量
        max_depth: 树的最大深度
        min_samples_split: 分裂内部节点所需的最小样本数
    
    返回:
        训练好的随机森林模型
    """
    start_time = time.time()
    
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )
    
    rf.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f"随机森林模型训练完成，耗时: {training_time:.2f}秒")
    
    return rf, training_time

def train_neural_network(X_train, y_train, hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=1000, random_state=42, use_keras=True):
    """
    训练神经网络模型
    
    参数:
        X_train: 训练集特征
        y_train: 训练集标签
        hidden_layer_sizes: 隐藏层大小
        activation: 激活函数
        solver: 优化器
        max_iter: 最大迭代次数
        random_state: 随机种子
        use_keras: 是否使用Keras
    
    返回:
        训练好的神经网络模型和训练时间
    """
    logger.info("训练神经网络模型")
    
    start_time = time.time()
    
    # 获取类别数量
    n_classes = len(np.unique(y_train))
    
    # 强制使用Keras
    if use_keras:
        try:
            # 将标签转换为one-hot编码
            y_train_onehot = to_categorical(y_train, num_classes=n_classes)
            
            # 使用函数式API创建模型
            inputs = Input(shape=(X_train.shape[1],))
            
            # 添加第一个隐藏层
            x = Dense(hidden_layer_sizes[0], activation=activation)(inputs)
            x = Dropout(0.2)(x)
            
            # 添加其他隐藏层
            for units in hidden_layer_sizes[1:]:
                x = Dense(units, activation=activation)(x)
                x = Dropout(0.2)(x)
            
            # 添加输出层
            outputs = Dense(n_classes, activation='softmax')(x)
            
            # 创建模型
            model = Model(inputs=inputs, outputs=outputs)
            
            # 编译模型
            model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
            
            # 训练模型
            model.fit(X_train, y_train_onehot, epochs=max_iter//10, batch_size=32, verbose=0)
            
            # 添加训练时间属性
            training_time = time.time() - start_time
            model.training_time = training_time
            
            logger.info(f"Keras神经网络模型训练完成，耗时: {training_time:.2f}秒")
            
            return model, training_time
        except Exception as e:
            logger.error(f"Keras模型训练失败: {e}")
            raise ImportError(f"无法使用Keras训练神经网络模型: {e}。请确保已安装tensorflow库: pip install tensorflow")
    
    # 如果明确指定不使用Keras，则使用scikit-learn的MLPClassifier
    logger.warning("不推荐使用scikit-learn的MLPClassifier，建议使用Keras模型")
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        max_iter=max_iter,
        random_state=random_state
    )
    
    model.fit(X_train, y_train)
    
    # 添加训练时间属性
    training_time = time.time() - start_time
    model.training_time = training_time
    
    logger.info(f"scikit-learn神经网络模型训练完成，耗时: {training_time:.2f}秒")
    
    return model, training_time

def optimize_model(X_train, y_train, model_type='svm', cv=5, n_iter=10):
    """
    使用网格搜索或随机搜索优化模型超参数
    
    参数:
        X_train: 训练集特征
        y_train: 训练集标签
        model_type: 模型类型，'knn', 'svm', 'rf', 或 'nn'
        cv: 交叉验证折数
        n_iter: 随机搜索迭代次数
    
    返回:
        最佳模型和最佳参数
    """
    start_time = time.time()
    
    if model_type == 'knn':
        model = KNeighborsClassifier()
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski']
        }
    
    elif model_type == 'svm':
        model = SVC(probability=True)
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'gamma': ['scale', 'auto', 0.1, 0.01]
        }
    
    elif model_type == 'rf':
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    
    elif model_type == 'nn':
        model = MLPClassifier(random_state=42, max_iter=1000)
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
            'activation': ['relu', 'tanh', 'logistic'],
            'solver': ['adam', 'sgd'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive']
        }
    
    # 使用随机搜索而不是网格搜索，以减少计算时间
    search = RandomizedSearchCV(
        model, param_grid, n_iter=n_iter, cv=cv, scoring='accuracy', n_jobs=-1, random_state=42
    )
    
    search.fit(X_train, y_train)
    
    optimization_time = time.time() - start_time
    print(f"模型优化完成，耗时: {optimization_time:.2f}秒")
    print(f"最佳参数: {search.best_params_}")
    print(f"最佳交叉验证得分: {search.best_score_:.4f}")
    
    return search.best_estimator_, search.best_params_, optimization_time

def save_model(model, file_path):
    """
    保存训练好的模型
    
    参数:
        model: 训练好的模型
        file_path: 保存路径
    """
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"模型已保存到: {file_path}")
    except Exception as e:
        print(f"保存模型失败: {e}")

def save_keras_model(model, save_path):
    """
    保存Keras模型
    
    参数:
        model: 训练好的Keras模型
        save_path: 保存路径
    """
    try:
        # 检查是否为Keras模型
        if 'keras' in str(type(model)).lower():
            # 确保目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # 确保文件路径有正确的扩展名
            if not (save_path.endswith('.keras') or save_path.endswith('.h5')):
                save_path = f"{save_path}.keras"  # 添加.keras扩展名
                logger.info(f"添加.keras扩展名到保存路径: {save_path}")
            
            # 保存模型架构和权重
            model.save(save_path)
            
            # 保存模型配置（如果可能）
            try:
                model_config = model.get_config()
                config_path = save_path.replace('.keras', '_config.json').replace('.h5', '_config.json')
                with open(config_path, 'w') as f:
                    json.dump(model_config, f)
            except Exception as e:
                logger.warning(f"无法保存模型配置: {e}")
            
            # 保存模型权重
            weights_path = save_path.replace('.keras', '_weights.h5').replace('.h5', '_weights.h5')
            model.save_weights(weights_path)
            
            logger.info(f"Keras模型已保存到: {save_path}")
            return True
        else:
            logger.warning(f"模型不是Keras模型，无法使用save_keras_model函数保存")
            return False
    except Exception as e:
        logger.error(f"保存Keras模型失败: {e}")
        return False
