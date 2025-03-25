"""
VOC气体分类 - 可视化模块

提供数据可视化功能
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import learning_curve
from logger.logger import Logger

# 创建日志记录器
logger = Logger('Figure')

# 设置字体
try:
    # 添加Times字体
    font_path = 'times.ttf'  # 字体文件在根目录下
    if os.path.exists(font_path):
        times_font = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = 'Times New Roman'
        logger.info("已加载Times New Roman字体")
    else:
        logger.warning(f"未找到字体文件: {font_path}")
except Exception as e:
    logger.error(f"加载字体失败: {e}")

# 设置图表样式
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

def plot_feature_distribution(data, feature, hue=None, title=None, save_path=None):
    """
    绘制特征分布图
    
    参数:
        data: 数据集DataFrame
        feature: 要绘制的特征名称
        hue: 分类变量
        title: 图表标题
        save_path: 保存路径
    """
    plt.figure(figsize=(12, 6))
    
    if hue is not None and hue in data.columns:
        sns.histplot(data=data, x=feature, hue=hue, kde=True)
    else:
        sns.histplot(data=data[feature], kde=True)
    
    if title:
        plt.title(title)
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.tight_layout()
    
    if save_path:
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        logger.info(f"特征分布图已保存到: {save_path}")
    
    plt.close()

def plot_correlation_matrix(data, title=None, save_path=None):
    """
    绘制相关性矩阵
    
    参数:
        data: 数据集DataFrame
        title: 图表标题
        save_path: 保存路径
    """
    # 计算相关性矩阵
    corr = data.corr()
    
    # 绘制热图
    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    
    if title:
        plt.title(title)
    plt.tight_layout()
    
    if save_path:
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        logger.info(f"相关性矩阵已保存到: {save_path}")
    
    plt.close()

def plot_pca_visualization(X, y, title=None, save_path=None):
    """
    使用PCA进行降维可视化
    
    参数:
        X: 特征矩阵
        y: 标签
        title: 图表标题
        save_path: 保存路径
    """
    # 使用PCA降维到2维
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # 创建DataFrame以便使用seaborn绘图
    df = pd.DataFrame({
        'PCA1': X_pca[:, 0],
        'PCA2': X_pca[:, 1],
        'Class': y
    })
    
    # 绘制散点图
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Class', palette='viridis', s=100, alpha=0.7)
    
    if title:
        plt.title(title)
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%})')
    plt.legend(title='Class')
    plt.tight_layout()
    
    if save_path:
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        logger.info(f"PCA可视化已保存到: {save_path}")
    
    plt.close()
    
    return X_pca

def plot_tsne_visualization(X, y, title=None, save_path=None):
    """
    使用t-SNE进行降维可视化
    
    参数:
        X: 特征矩阵
        y: 标签
        title: 图表标题
        save_path: 保存路径
    """
    # 使用t-SNE降维到2维
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)
    
    # 创建DataFrame以便使用seaborn绘图
    df = pd.DataFrame({
        't-SNE1': X_tsne[:, 0],
        't-SNE2': X_tsne[:, 1],
        'Class': y
    })
    
    # 绘制散点图
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='t-SNE1', y='t-SNE2', hue='Class', palette='viridis', s=100, alpha=0.7)
    
    if title:
        plt.title(title)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(title='Class')
    plt.tight_layout()
    
    if save_path:
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        logger.info(f"t-SNE可视化已保存到: {save_path}")
    
    plt.close()
    
    return X_tsne

def plot_feature_importance(feature_names, importances, title=None, top_n=20, save_path=None):
    """
    绘制特征重要性图
    
    参数:
        feature_names: 特征名称列表
        importances: 特征重要性列表
        title: 图表标题
        top_n: 显示前n个重要特征
        save_path: 保存路径
    """
    # 获取特征重要性排序
    indices = np.argsort(importances)[::-1]
    top_indices = indices[:top_n]
    
    # 绘制条形图
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(top_indices)), importances[top_indices], align='center')
    plt.yticks(range(len(top_indices)), [feature_names[i] for i in top_indices])
    plt.gca().invert_yaxis()  # 从上到下显示重要性递减
    
    if title:
        plt.title(title)
    plt.xlabel('Feature Importance')
    plt.tight_layout()
    
    if save_path:
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        logger.info(f"特征重要性图已保存到: {save_path}")
    
    plt.close()

def plot_class_distribution(y, title=None, save_path=None):
    """
    绘制类别分布图
    
    参数:
        y: 类别标签
        title: 图表标题
        save_path: 保存路径
    """
    # 计算类别分布
    class_counts = pd.Series(y).value_counts().sort_index()
    
    # 绘制条形图
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x=class_counts.index, y=class_counts.values)
    
    # 在条形上方显示数值
    for i, count in enumerate(class_counts.values):
        ax.text(i, count + 0.1, str(count), ha='center')
    
    if title:
        plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        logger.info(f"类别分布图已保存到: {save_path}")
    
    plt.close()