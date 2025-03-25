"""
VOC气体分类系统

提供基于机器学习的VOC气体分类功能
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from logger.logger import Logger
from .preprocess import (
    load_data, explore_data, preprocess_data, 
    select_features, visualize_feature_importance
)
from .train import (
    train_knn, train_svm, train_random_forest, train_neural_network,
    optimize_model, save_model, save_keras_model
)
from .evaluate import (
    evaluate_model, plot_confusion_matrix, plot_roc_curve,
    plot_precision_recall_curve, plot_learning_curve, compare_models,
    plot_keras_learning_curve
)
from .predict import (
    load_model, predict_single_sample, predict_batch,
    create_prediction_report, save_prediction_report
)
from .figure import (
    plot_feature_distribution, plot_correlation_matrix,
    plot_pca_visualization, plot_tsne_visualization,
    plot_feature_importance, plot_class_distribution
)

__version__ = '1.0.0'

class Classification:
    """VOC气体分类模型类"""
    
    def __init__(self, input_csv, feature_selection="rf_importance", k_features=250, target_column="gas"):
        """
        初始化分类模型
        
        参数:
            input_csv: 输入CSV文件路径
            feature_selection: 特征选择方法，可选 "rf_importance", "selectkbest", "rfe", "none"
            k_features: 选择的特征数量
            target_column: 目标变量列名
        """
        # 初始化日志记录器
        self.logger = Logger('Classification')
        self.input_csv = input_csv
        self.feature_selection = feature_selection
        self.k_features = k_features
        self.target_column = target_column
        
        # 创建输出目录
        self.output_dirs = ['models', 'results', 'figures']
        for dir_name in self.output_dirs:
            os.makedirs(dir_name, exist_ok=True)
        
        # 初始化属性
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.feature_names = None
        self.selected_indices = None
        self.selected_feature_names = None
        self.models = {}
        self.metrics = {}
        self.best_model = None
        self.best_model_name = None
        
        self.logger.info(f"初始化VOC气体分类模型，特征选择方法: {feature_selection}，特征数量: {k_features}")
    
    def load_dataset(self):
        """加载数据集"""
        self.logger.info(f"加载数据集: {self.input_csv}")
        self.data = load_data(self.input_csv)
        if self.data is None:
            self.logger.error("数据集加载失败")
            return False
        self.logger.info("数据集加载成功")
        return True
    
    def analyze_dataset(self):
        """分析数据集"""
        if self.data is None:
            self.logger.error("请先加载数据集")
            return
        
        self.logger.info("进行数据集探索性分析")
        stats = explore_data(self.data)
        
        # 绘制类别分布
        self.logger.info("绘制类别分布图")
        plot_class_distribution(
            self.data[self.target_column], 
            title="VOC Gas Class Distribution", 
            save_path="figures/class_distribution.png"
        )
        
        # 绘制传感器相关性矩阵
        self.logger.info("绘制传感器相关性矩阵")
        # 获取传感器列表
        sensors = ['TGS2603', 'TGS2630', 'TGS813', 'TGS822', 'MQ-135', 'MQ-137', 'MQ-138', '2M012', 'VOCS-P', '2SH12']
        
        # 计算每个传感器的平均响应
        sensor_means = {}
        for sensor in sensors:
            # 获取当前传感器的所有响应列
            sensor_columns = [col for col in self.data.columns if col.startswith(f"{sensor}_")]
            if sensor_columns:
                # 计算平均响应
                sensor_means[sensor] = self.data[sensor_columns].mean(axis=1)
        
        # 创建传感器平均响应的DataFrame
        if sensor_means:
            sensor_df = pd.DataFrame(sensor_means)
            
            # 绘制传感器相关性矩阵
            plot_correlation_matrix(
                sensor_df, 
                title="Sensor Response Correlation Matrix", 
                save_path="figures/sensor_correlation_matrix.png"
            )
            
            # 为每个传感器绘制单独的分布图
            self.logger.info("绘制传感器特征分布图")
            for sensor in sensors:
                if sensor in sensor_df.columns:
                    plt.figure(figsize=(10, 6))
                    sns.histplot(data=sensor_df, x=sensor, hue=self.data[self.target_column], kde=True)
                    plt.title(f"{sensor} Feature Distribution")
                    plt.xlabel("Average")
                    plt.ylabel("Count")
                    plt.tight_layout()
                    
                    # 保存图片
                    save_path = f"figures/{sensor}_feature_distribution.png"
                    plt.savefig(save_path, dpi=300)
                    plt.close()
                    self.logger.info(f"传感器特征分布图已保存到: {save_path}")
        else:
            self.logger.warning("未找到传感器数据，无法绘制相关性矩阵和分布图")
        
        self.logger.info("数据集分析完成")
    
    def preprocess(self):
        """数据预处理"""
        if self.data is None:
            self.logger.error("请先加载数据集")
            return False
        
        self.logger.info("开始数据预处理")
        
        # 数据预处理，指定test_size=0.4
        preprocessed_data = preprocess_data(self.data, test_size=0.4)
        
        # 检查返回值类型
        if isinstance(preprocessed_data, tuple):
            # 如果返回的是元组，解包元组
            self.X_train, self.X_test, self.y_train, self.y_test, self.scaler, self.feature_names, self.class_names = preprocessed_data
            
            # 获取已经缩放的数据
            self.X_train_scaled = self.X_train
            self.X_test_scaled = self.X_test
        else:
            # 如果返回的是DataFrame，按照原计划处理
            self.data = preprocessed_data
            
            # 划分特征和标签
            X = self.data.drop(columns=[self.target_column])
            y = self.data[self.target_column]
            
            # 划分训练集和测试集
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.4, random_state=42, stratify=y
            )
            
            # 特征缩放
            self.scaler = StandardScaler()
            self.X_train_scaled = self.scaler.fit_transform(self.X_train)
            self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # 特征选择
        self.logger.info(f"使用 {self.feature_selection} 方法选择 {self.k_features} 个特征")
        
        # 确保feature_names存在
        if not hasattr(self, 'feature_names') or self.feature_names is None:
            if hasattr(self, 'X_train') and hasattr(self.X_train, 'columns'):
                self.feature_names = self.X_train.columns
            else:
                self.feature_names = [f"feature_{i}" for i in range(self.X_train_scaled.shape[1])]
        
        self.selected_indices, self.selected_feature_names = select_features(
            self.X_train_scaled, self.y_train, 
            method=self.feature_selection, 
            k=self.k_features,
            feature_names=self.feature_names
        )
        
        # 获取选择的特征
        self.X_train_selected = self.X_train_scaled[:, self.selected_indices]
        self.X_test_selected = self.X_test_scaled[:, self.selected_indices]
        
        self.logger.info(f"数据预处理完成，训练集大小: {self.X_train_scaled.shape}, 测试集大小: {self.X_test_scaled.shape}")
        self.logger.info(f"选择的特征数量: {len(self.selected_indices)}")
        
        return True
    
    def train_models(self, optimize=False):
        """训练模型"""
        if self.X_train_selected is None or self.y_train is None:
            self.logger.error("请先进行数据预处理")
            return False
        
        self.logger.info("开始训练模型")
        
        # 训练KNN模型
        self.logger.info("训练KNN模型")
        if optimize:
            knn, best_params = optimize_model(
                self.X_train_selected, self.y_train, 
                model_type='knn', 
                param_grid={'n_neighbors': [3, 5, 7, 9, 11]}
            )
            self.logger.info(f"KNN最佳参数: {best_params}")
        else:
            knn, _ = train_knn(self.X_train_selected, self.y_train)
        
        # 评估KNN模型
        knn_metrics = evaluate_model(knn, self.X_test_selected, self.y_test)
        
        # 保存模型
        save_model(knn, "models/knn_model.pkl")
        
        self.models['KNN'] = knn
        self.metrics['KNN'] = knn_metrics
        
        # 训练SVM模型
        self.logger.info("训练SVM模型")
        if optimize:
            svm, best_params = optimize_model(
                self.X_train_selected, self.y_train, 
                model_type='svm', 
                param_grid={'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto', 0.1, 0.01]}
            )
            self.logger.info(f"SVM最佳参数: {best_params}")
        else:
            svm, _ = train_svm(self.X_train_selected, self.y_train)
        
        # 评估SVM模型
        svm_metrics = evaluate_model(svm, self.X_test_selected, self.y_test)
        
        # 保存模型
        save_model(svm, "models/svm_model.pkl")
        
        self.models['SVM'] = svm
        self.metrics['SVM'] = svm_metrics
        
        # 训练随机森林模型
        self.logger.info("训练随机森林模型")
        if optimize:
            rf, best_params = optimize_model(
                self.X_train_selected, self.y_train, 
                model_type='rf', 
                param_grid={'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30]}
            )
            self.logger.info(f"随机森林最佳参数: {best_params}")
        else:
            rf, _ = train_random_forest(self.X_train_selected, self.y_train)
        
        # 评估随机森林模型
        rf_metrics = evaluate_model(rf, self.X_test_selected, self.y_test)
        
        # 保存模型
        save_model(rf, "models/rf_model.pkl")
        
        self.models['RF'] = rf
        self.metrics['RF'] = rf_metrics
        
        # 训练神经网络模型
        self.logger.info("训练神经网络模型")
        if optimize:
            nn, best_params = optimize_model(
                self.X_train_selected, self.y_train, 
                model_type='nn', 
                param_grid={'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)]}
            )
            self.logger.info(f"神经网络最佳参数: {best_params}")
        else:
            try:
                # 强制使用Keras
                nn, _ = train_neural_network(self.X_train_selected, self.y_train, use_keras=True)
            except ImportError as e:
                self.logger.error(f"{e}")
                self.logger.error("神经网络模型训练失败，跳过此模型")
                nn = None
        
        # 如果神经网络模型训练成功
        if nn is not None:
            # 评估神经网络模型
            nn_metrics = evaluate_model(nn, self.X_test_selected, self.y_test)
            
            # 保存模型
            save_model(nn, "models/nn_model.pkl")
            
            # 如果是Keras模型，使用专门的函数保存
            if 'keras' in str(type(nn)).lower():
                save_keras_model(nn, "models/nn_model")
            
            self.models['NN'] = nn
            self.metrics['NN'] = nn_metrics
        else:
            self.logger.warning("神经网络模型未训练，将不包含在最终结果中")
        
        # 找出最佳模型
        best_accuracy = 0
        for model_name, metrics in self.metrics.items():
            if metrics['accuracy'] > best_accuracy:
                best_accuracy = metrics['accuracy']
                self.best_model_name = model_name
                self.best_model = self.models[model_name]
        
        self.logger.info(f"模型训练完成，最佳模型: {self.best_model_name}，准确率: {best_accuracy:.4f}")
        return True
    
    def evaluate(self):
        """评估模型"""
        if not self.models:
            self.logger.error("请先训练模型")
            return False
        
        self.logger.info("开始评估模型")
        
        # 绘制混淆矩阵
        for model_name, model in self.models.items():
            self.logger.info(f"绘制{model_name}混淆矩阵")
            try:
                plot_confusion_matrix(
                    model, self.X_test_selected, self.y_test,
                    title=f"{model_name} Confusion Matrix",
                    save_path=f"figures/{model_name}_confusion_matrix.png"
                )
            except Exception as e:
                self.logger.error(f"绘制{model_name}混淆矩阵失败: {e}")
        
        # 绘制ROC曲线和精确率-召回率曲线
        # 过滤掉Keras模型，因为它们需要特殊处理
        sklearn_models = {name: model for name, model in self.models.items() 
                         if 'keras' not in str(type(model)).lower()}
        
        if sklearn_models:
            # 绘制ROC曲线
            self.logger.info("绘制ROC曲线")
            try:
                plot_roc_curve(
                    sklearn_models, self.X_test_selected, self.y_test,
                    title="VOC Gas Classification ROC Curve",
                    save_path="figures/roc_curve.png"
                )
            except Exception as e:
                self.logger.error(f"绘制ROC曲线失败: {e}")
            
            # 绘制精确率-召回率曲线
            self.logger.info("绘制精确率-召回率曲线")
            try:
                plot_precision_recall_curve(
                    sklearn_models, self.X_test_selected, self.y_test,
                    title="VOC Gas Classification Precision-Recall Curve",
                    save_path="figures/precision_recall_curve.png"
                )
            except Exception as e:
                self.logger.error(f"绘制精确率-召回率曲线失败: {e}")
        
        # 绘制学习曲线
        self.logger.info("绘制学习曲线")
        for model_name, model in self.models.items():
            # 对于Keras模型，使用专门的函数绘制学习曲线
            if 'keras' in str(type(model)).lower():
                self.logger.info(f"使用Keras专用函数绘制{model_name}学习曲线")
                try:
                    plot_keras_learning_curve(
                        model, self.X_train_selected, self.y_train,
                        title=f"{model_name} Learning Curve",
                        epochs=50,  # 可以根据需要调整
                        batch_size=32,
                        validation_split=0.2,
                        save_path=f"figures/{model_name}_learning_curve.png"
                    )
                except Exception as e:
                    self.logger.error(f"绘制{model_name}学习曲线失败: {e}")
            else:
                # 对于scikit-learn模型，使用原来的函数
                try:
                    plot_learning_curve(
                        model, self.X_train_selected, self.y_train,
                        title=f"{model_name} Learning Curve",
                        save_path=f"figures/{model_name}_learning_curve.png"
                    )
                except Exception as e:
                    self.logger.error(f"绘制{model_name}学习曲线失败: {e}")
        
        # 比较模型性能
        self.logger.info("比较模型性能")
        try:
            compare_models(
                self.metrics,
                title="VOC Gas Classification Model Comparison",
                save_path="figures/model_comparison.png"
            )
        except Exception as e:
            self.logger.error(f"比较模型性能失败: {e}")
        
        self.logger.info("模型评估完成")
        return True
    
    def predict(self, samples):
        """使用最佳模型进行预测"""
        if self.best_model is None:
            self.logger.error("请先训练模型")
            return None
        
        self.logger.info(f"使用{self.best_model_name}模型进行预测")
        
        predicted_classes, max_probs, probabilities = predict_batch(
            self.best_model, samples, 
            scaler=self.scaler, 
            feature_indices=self.selected_indices
        )
        
        self.logger.info("预测完成")
        return predicted_classes, max_probs, probabilities
    
    def run(self):
        """运行模型训练流水线"""
        self.logger.info("开始运行模型训练流水线")
        
        # 加载数据集
        if not self.load_dataset():
            return False
        
        # 数据预处理
        if not self.preprocess():
            return False
        
        # 训练模型
        if not self.train_models():
            return False
        
        self.logger.info("模型训练流水线运行完成")
        return True