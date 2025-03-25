"""
VOC 气体分类
"""

from logger import Logger
from classification import Classification


def voc_classification():
    logger = Logger('Classification')
    # VOC 气体分类模型
    logger.info("开始 VOC 气体分类模型训练")
    cla = Classification(
        input_csv="gsalc.csv",
        feature_selection="rf_importance",  # 使用随机森林特征重要性
        k_features=250  # 选择前 250 个重要特征
    )

    # 运行模型训练流水线
    logger.info("运行气体分类模型训练流水线")
    cla.run()

    # 分析数据集
    logger.info("分析气体分类数据集")
    cla.analyze_dataset()

    # 评估模型
    logger.info("评估气体分类模型")
    cla.evaluate()
    logger.info("完成 VOC 气体分类模型训练")

if __name__ == "__main__":
    voc_classification()