"""
VOC 气体分类 - 评估模块

提供模型评估、结果分析和可视化功能
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.model_selection import learning_curve, validation_curve
import time
import os

def evaluate_model(model, X_test, y_test):
    """
    评估模型性能
    
    参数:
        model: 训练好的模型
        X_test: 测试集特征
        y_test: 测试集标签
    
    返回:
        包含各种评估指标的字典
    """
    start_time = time.time()
    
    # 检查是否为Keras模型
    is_keras_model = 'keras' in str(type(model)).lower()
    
    # 预测
    if is_keras_model:
        # Keras模型返回概率
        y_prob = model.predict(X_test)
        # 将概率转换为类别
        y_pred = np.argmax(y_prob, axis=1)
        # 如果y_test不是one-hot编码，则不需要转换
        if len(y_test.shape) > 1 and y_test.shape[1] > 1:
            y_test_labels = np.argmax(y_test, axis=1)
        else:
            y_test_labels = y_test
    else:
        # 普通模型直接预测类别
        y_pred = model.predict(X_test)
        y_test_labels = y_test
        
        # 如果模型支持概率预测
        try:
            y_prob = model.predict_proba(X_test)
        except:
            y_prob = None
    
    # 计算评估指标
    accuracy = accuracy_score(y_test_labels, y_pred)
    precision = precision_score(y_test_labels, y_pred, average='weighted')
    recall = recall_score(y_test_labels, y_pred, average='weighted')
    f1 = f1_score(y_test_labels, y_pred, average='weighted')
    
    # 混淆矩阵
    cm = confusion_matrix(y_test_labels, y_pred)
    
    # 分类报告
    report = classification_report(y_test_labels, y_pred)
    
    evaluation_time = time.time() - start_time
    
    # 输出评估结果
    print(f"模型评估完成，耗时: {evaluation_time:.2f}秒")
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1:.4f}")
    print("\n分类报告:")
    print(report)
    
    # 返回评估指标
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'classification_report': report,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'evaluation_time': evaluation_time
    }
    
    return metrics

def plot_confusion_matrix(model, X_test, y_test, title=None, save_path=None):
    """
    绘制混淆矩阵
    
    参数:
        model: 训练好的模型
        X_test: 测试集特征
        y_test: 测试集标签
        title: 图表标题
        save_path: 保存路径
    """
    # 检查是否为Keras模型
    is_keras_model = 'keras' in str(type(model)).lower()
    
    # 使用模型进行预测
    if is_keras_model:
        # Keras模型返回概率
        y_prob = model.predict(X_test)
        # 将概率转换为类别
        y_pred = np.argmax(y_prob, axis=1)
        # 如果y_test不是one-hot编码，则不需要转换
        if len(y_test.shape) > 1 and y_test.shape[1] > 1:
            y_test = np.argmax(y_test, axis=1)
    else:
        # 普通模型直接预测类别
        y_pred = model.predict(X_test)
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    
    # 归一化混淆矩阵
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # 获取类别名称
    if hasattr(model, 'classes_'):
        class_names = model.classes_
    else:
        class_names = sorted(np.unique(y_test))
    
    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    
    if title:
        plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    if save_path:
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()

def plot_roc_curve(models, X_test, y_test, title=None, save_path=None):
    """
    绘制ROC曲线
    
    参数:
        models: 训练好的模型字典 {model_name: model}
        X_test: 测试集特征
        y_test: 测试集标签
        title: 图表标题
        save_path: 保存路径
    """
    plt.figure(figsize=(10, 8))
    
    # 获取类别名称
    class_names = sorted(np.unique(y_test))
    n_classes = len(class_names)
    
    # 为每个模型绘制ROC曲线
    for model_name, model in models.items():
        # 获取预测概率
        try:
            y_prob = model.predict_proba(X_test)
            
            # 计算每个类别的ROC曲线
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            
            # 将标签转换为one-hot编码
            y_test_bin = np.zeros((len(y_test), n_classes))
            for i, label in enumerate(y_test):
                y_test_bin[i, label] = 1
            
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            
            # 计算微平均ROC曲线
            fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_prob.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            
            # 绘制微平均ROC曲线
            plt.plot(fpr["micro"], tpr["micro"],
                     label=f'{model_name} (AUC = {roc_auc["micro"]:.2f})',
                     lw=2)
        except:
            print(f"模型 {model_name} 不支持概率预测，跳过ROC曲线绘制")
    
    # 绘制对角线
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    if title:
        plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    if save_path:
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()

def plot_precision_recall_curve(models, X_test, y_test, title=None, save_path=None):
    """
    绘制精确率-召回率曲线
    
    参数:
        models: 训练好的模型字典 {model_name: model}
        X_test: 测试集特征
        y_test: 测试集标签
        title: 图表标题
        save_path: 保存路径
    """
    plt.figure(figsize=(10, 8))
    
    # 获取类别名称
    class_names = sorted(np.unique(y_test))
    n_classes = len(class_names)
    
    # 为每个模型绘制精确率-召回率曲线
    for model_name, model in models.items():
        # 获取预测概率
        try:
            y_prob = model.predict_proba(X_test)
            
            # 计算每个类别的精确率-召回率曲线
            precision = dict()
            recall = dict()
            avg_precision = dict()
            
            # 将标签转换为one-hot编码
            y_test_bin = np.zeros((len(y_test), n_classes))
            for i, label in enumerate(y_test):
                y_test_bin[i, label] = 1
            
            # 计算微平均精确率-召回率曲线
            precision["micro"], recall["micro"], _ = precision_recall_curve(
                y_test_bin.ravel(), y_prob.ravel()
            )
            avg_precision["micro"] = average_precision_score(
                y_test_bin.ravel(), y_prob.ravel()
            )
            
            # 绘制微平均精确率-召回率曲线
            plt.plot(recall["micro"], precision["micro"],
                     label=f'{model_name} (AP = {avg_precision["micro"]:.2f})',
                     lw=2)
        except:
            print(f"模型 {model_name} 不支持概率预测，跳过精确率-召回率曲线绘制")
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    if title:
        plt.title(title)
    plt.legend(loc="lower left")
    plt.tight_layout()
    
    if save_path:
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()

def plot_learning_curve(estimator, X, y, title=None, cv=None, train_sizes=np.linspace(0.1, 1.0, 5), save_path=None):
    """
    绘制学习曲线
    
    参数:
        estimator: 模型
        X: 特征
        y: 标签
        title: 图表标题
        cv: 交叉验证折数，如果为None则根据数据集大小自动选择
        train_sizes: 训练集大小比例
        save_path: 保存路径
    """
    # 检查类别数量
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        # 如果只有一个类别，创建一个错误信息图
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "Error: Cannot plot learning curve with only one class", 
                 horizontalalignment='center', verticalalignment='center',
                 transform=plt.gca().transAxes)
        if title:
            plt.title(title)
        plt.tight_layout()
        
        if save_path:
            # 确保目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300)
            plt.close()
        return
    
    plt.figure(figsize=(10, 6))
    
    # 检查数据集大小，调整交叉验证折数
    if cv is None:
        # 获取每个类别的样本数
        _, class_counts = np.unique(y, return_counts=True)
        min_samples = min(class_counts)
        
        # 如果最小类别样本数小于5，使用留一法交叉验证
        if min_samples < 5:
            from sklearn.model_selection import LeaveOneOut
            cv = LeaveOneOut()
            print(f"警告: 数据集中最小类别样本数为{min_samples}，使用留一法交叉验证")
        else:
            # 否则使用最小类别样本数和5中的较小值作为折数
            cv = min(5, min_samples)
            print(f"使用{cv}折交叉验证")
    
    # 对于KNN模型，确保n_neighbors不大于最小类别样本数
    if hasattr(estimator, 'n_neighbors'):
        _, class_counts = np.unique(y, return_counts=True)
        min_samples = min(class_counts)
        
        # 计算每个类别在每个折中的最小样本数
        if isinstance(cv, int):
            min_samples_per_fold = min_samples // cv
        else:
            min_samples_per_fold = min_samples
        
        # 如果n_neighbors大于每个折中的最小样本数，则调整n_neighbors
        if estimator.n_neighbors > min_samples_per_fold:
            print(f"警告: KNN的n_neighbors({estimator.n_neighbors})大于每个折中的最小样本数({min_samples_per_fold})，临时调整为{max(1, min_samples_per_fold)}")
            # 临时创建一个新的估计器，避免修改原始估计器
            from sklearn.neighbors import KNeighborsClassifier
            # 确保n_neighbors至少为1
            estimator = KNeighborsClassifier(n_neighbors=max(1, min_samples_per_fold))
    
    # 对于SVM模型，检查是否有足够的样本
    if hasattr(estimator, '_impl') and 'SVC' in str(type(estimator)):
        # 检查每个类别的样本数
        _, class_counts = np.unique(y, return_counts=True)
        if min(class_counts) < 2:
            # 如果某个类别的样本数小于2，创建一个错误信息图
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, f"Error: SVM requires at least 2 samples per class, but found {min(class_counts)}", 
                     horizontalalignment='center', verticalalignment='center',
                     transform=plt.gca().transAxes)
            if title:
                plt.title(title)
            plt.tight_layout()
            
            if save_path:
                # 确保目录存在
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300)
                plt.close()
            return
    
    try:
        # 使用更安全的参数设置
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, train_sizes=train_sizes, 
            scoring='accuracy', n_jobs=1,  # 使用单线程以避免并发问题
            error_score=np.nan  # 使用NaN而不是引发错误
        )
        
        # 过滤掉NaN值
        valid_indices = ~np.isnan(train_scores).any(axis=1) & ~np.isnan(test_scores).any(axis=1)
        if not valid_indices.any():
            raise ValueError("All scores are NaN, cannot plot learning curve")
        
        train_sizes = train_sizes[valid_indices]
        train_scores = train_scores[valid_indices]
        test_scores = test_scores[valid_indices]
        
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training Score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Validation Score")
        
        plt.xlabel("Training Examples")
        plt.ylabel("Accuracy")
        if title:
            plt.title(title)
        plt.legend(loc="best")
        plt.grid(True)
        plt.tight_layout()
        
        if save_path:
            # 确保目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300)
            plt.close()
    except Exception as e:
        plt.close()
        print(f"绘制学习曲线时出错: {str(e)}")
        # 创建一个简单的错误信息图
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f"Error: {str(e)}", 
                 horizontalalignment='center', verticalalignment='center',
                 transform=plt.gca().transAxes)
        if title:
            plt.title(title)
        plt.tight_layout()
        
        if save_path:
            # 确保目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300)
            plt.close()

def compare_models(metrics, title=None, save_path=None):
    """
    比较不同模型的性能
    
    参数:
        metrics: 评估指标字典 {model_name: metrics}
        title: 图表标题
        save_path: 保存路径
    """
    # 提取模型名称和评估指标
    model_names = list(metrics.keys())
    metrics_list = list(metrics.values())
    
    # 提取评估指标
    accuracies = [m['accuracy'] for m in metrics_list]
    precisions = [m['precision'] for m in metrics_list]
    recalls = [m['recall'] for m in metrics_list]
    f1_scores = [m['f1'] for m in metrics_list]
    
    # 创建比较表格
    comparison_df = pd.DataFrame({
        'Model': model_names,
        'Accuracy': accuracies,
        'Precision': precisions,
        'Recall': recalls,
        'F1 Score': f1_scores
    })
    
    # 绘制性能比较图
    plt.figure(figsize=(12, 8))
    
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    bar_width = 0.2
    index = np.arange(len(model_names))
    
    for i, metric in enumerate(metrics_to_plot):
        plt.bar(index + i * bar_width, comparison_df[metric], bar_width,
                label=metric)
    
    plt.xlabel('Model')
    plt.ylabel('Score')
    if title:
        plt.title(title)
    plt.xticks(index + bar_width * (len(metrics_to_plot) - 1) / 2, model_names)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    
    return comparison_df

def plot_keras_learning_curve(model, X, y, title=None, epochs=100, batch_size=32, validation_split=0.2, save_path=None):
    """
    绘制Keras模型的学习曲线
    
    参数:
        model: 训练好的Keras模型
        X: 特征
        y: 标签
        title: 图表标题
        epochs: 训练轮数
        batch_size: 批量大小
        validation_split: 验证集比例
        save_path: 保存路径
    """
    from keras.api.utils import to_categorical
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    
    # 检查是否为Keras模型
    if 'keras' not in str(type(model)).lower():
        raise ValueError("只能为Keras模型绘制Keras学习曲线")
    
    # 获取类别数量
    n_classes = len(np.unique(y))
    
    # 将标签转换为one-hot编码
    y_onehot = to_categorical(y, num_classes=n_classes)
    
    # 重新编译模型，确保它有正确的损失函数和指标
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # 训练模型并记录历史
    history = model.fit(
        X, y_onehot,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=0
    )
    
    # 绘制学习曲线
    plt.figure(figsize=(12, 5))
    
    # 绘制训练和验证准确率
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    # 绘制训练和验证损失
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    if title:
        plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    
    return history