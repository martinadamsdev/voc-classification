"""
VOC气体分类 - 预测模块

实现气体类型分类功能，使用纯函数式风格
"""

import pickle
import numpy as np
import pandas as pd

def load_model(model_path):
    """
    加载训练好的模型
    
    参数:
        model_path: 模型文件路径
    
    返回:
        加载的模型
    """
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"模型已从 {model_path} 加载")
        return model
    except Exception as e:
        print(f"加载模型失败: {e}")
        return None

def predict_single_sample(model, sample, scaler=None, feature_indices=None):
    """
    对单个样本进行预测
    
    参数:
        model: 训练好的模型
        sample: 待预测的样本
        scaler: 特征缩放器
        feature_indices: 特征选择后的特征索引
    
    返回:
        预测的类别和概率
    """
    # 确保样本是二维数组
    if len(sample.shape) == 1:
        sample = sample.reshape(1, -1)
    
    # 应用特征缩放
    if scaler is not None:
        sample = scaler.transform(sample)
    
    # 应用特征选择
    if feature_indices is not None:
        sample = sample[:, feature_indices]
    
    # 预测类别
    predicted_class = model.predict(sample)[0]
    
    # 预测概率（如果模型支持）
    try:
        probabilities = model.predict_proba(sample)[0]
        max_prob = np.max(probabilities)
    except:
        probabilities = None
        max_prob = None
    
    return predicted_class, max_prob, probabilities

def predict_batch(model, samples, scaler=None, feature_indices=None):
    """
    对一批样本进行预测
    
    参数:
        model: 训练好的模型
        samples: 待预测的样本批次
        scaler: 特征缩放器
        feature_indices: 特征选择后的特征索引
    
    返回:
        预测的类别和概率
    """
    # 应用特征缩放
    if scaler is not None:
        samples = scaler.transform(samples)
    
    # 应用特征选择
    if feature_indices is not None:
        samples = samples[:, feature_indices]
    
    # 预测类别
    predicted_classes = model.predict(samples)
    
    # 预测概率（如果模型支持）
    try:
        probabilities = model.predict_proba(samples)
        max_probs = np.max(probabilities, axis=1)
    except:
        probabilities = None
        max_probs = None
    
    return predicted_classes, max_probs, probabilities

def create_prediction_report(samples, true_labels, predicted_classes, probabilities=None, class_names=None):
    """
    创建预测报告
    
    参数:
        samples: 样本数据
        true_labels: 真实标签
        predicted_classes: 预测的类别
        probabilities: 预测的概率
        class_names: 类别名称
    
    返回:
        预测报告DataFrame
    """
    # 创建基本报告
    report = pd.DataFrame({
        '真实标签': true_labels,
        '预测标签': predicted_classes,
        '预测正确': true_labels == predicted_classes
    })
    
    # 如果有概率信息，添加到报告中
    if probabilities is not None:
        # 添加最大概率
        report['预测置信度'] = np.max(probabilities, axis=1)
        
        # 添加每个类别的概率
        if class_names is not None:
            for i, class_name in enumerate(class_names):
                report[f'{class_name}概率'] = probabilities[:, i]
    
    # 计算准确率
    accuracy = (report['预测正确'].sum() / len(report)) * 100
    print(f"预测准确率: {accuracy:.2f}%")
    
    return report

def save_prediction_report(report, file_path):
    """
    保存预测报告
    
    参数:
        report: 预测报告DataFrame
        file_path: 保存路径
    """
    try:
        report.to_csv(file_path, index=False)
        print(f"预测报告已保存到: {file_path}")
    except Exception as e:
        print(f"保存预测报告失败: {e}")