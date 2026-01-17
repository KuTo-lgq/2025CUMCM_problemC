# 导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings('ignore')

# 检查 imbalanced-learn 库是否安装
try:
    import imblearn
    print(f"✓ imbalanced-learn 版本: {imblearn.__version__}")
except ImportError:
    print("✗ 未找到 imbalanced-learn 库。")
    print("请运行: pip install imbalanced-learn")
    raise

print("\n库导入完成")

def setup_chinese_font():
    """设置中文字体"""
    font_path = 'E:/Mathematics_Modeling_study/2025_CUMCM/fonts/HPSIMPLIFIEDHANS-REGULAR.TTF'
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        font_prop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = font_prop.get_name()
        plt.rcParams['axes.unicode_minus'] = False
        print(f"✓ 成功加载并设置字体: {font_prop.get_name()}")
        return font_prop
    else:
        for font_name in ['Microsoft YaHei', 'SimHei', 'PingFang SC']:
            if any(f.name == font_name for f in fm.fontManager.ttflist):
                plt.rcParams['font.family'] = font_name
                plt.rcParams['axes.unicode_minus'] = False
                print(f"✓ 使用系统字体: {font_name}")
                return fm.FontProperties(family=font_name)
    raise RuntimeError("未能找到可用的中文字体")

try:
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_style("whitegrid")
    font_prop = setup_chinese_font()
    print("✓ 字体设置成功")
except Exception as e:
    print(f"✗ 字体设置失败: {str(e)}")
    raise
# 加载并进行与原 notebook 相同的预处理
try:
    data_path = '../../Stem/C题/附件.xlsx'
    df_female = pd.read_excel(data_path, sheet_name=1)
    print(f"数据加载成功！女胎数据形状: {df_female.shape}")
except Exception as e:
    print(f"数据加载失败: {e}")
    # 备用路径
    try:
        data_path = '../Stem/C题/附件.xlsx'
        df_female = pd.read_excel(data_path, sheet_name=1)
        print(f"备用路径加载成功！女胎数据形状: {df_female.shape}")
    except Exception as e2:
        print(f"备用路径也失败: {e2}")
        raise

# 创建工作副本
df = df_female.copy()

# --- 与原 notebook 相同的预处理流程 ---

# 1. 目标变量创建
df['Is_Abnormal'] = df['染色体的非整倍体'].notna().astype(int)
df['Detailed_Abnormality'] = df['染色体的非整倍体'].fillna('Normal')

# 2. 缺失值处理
# 对于数值型特征，使用中位数填充
numeric_columns = df.select_dtypes(include=[np.number]).columns
for col in numeric_columns:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)

# 3. 特征工程
z_value_columns = ['13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值', 'X染色体的Z值']
for col in z_value_columns:
    df[f'abs_{col}'] = df[col].abs()
for col in z_value_columns:
    df[f'{col}_squared'] = df[col] ** 2
abs_z_columns = [f'abs_{col}' for col in z_value_columns]
df['Max_Abs_Z'] = df[abs_z_columns].max(axis=1)
df['Z_above_3_count'] = (df[abs_z_columns] > 3.0).sum(axis=1)
df['Z_mean'] = df[z_value_columns].mean(axis=1)
df['Z_std'] = df[z_value_columns].std(axis=1)

# 4. 准备建模特征
feature_columns = z_value_columns.copy()
derived_z_features = [f'abs_{col}' for col in z_value_columns] + \
                     [f'{col}_squared' for col in z_value_columns] + \
                     ['Max_Abs_Z', 'Z_above_3_count', 'Z_mean', 'Z_std']
feature_columns.extend(derived_z_features)

other_features = ['孕妇BMI', '年龄'] # 使用'孕妇BMI',替换'孕妇 BMI指标'
for feat in other_features:
    if feat in df.columns:
        feature_columns.append(feat)

gc_features = ['13号染色体的GC含量', '18号染色体的GC含量', '21号染色体的GC含量']
feature_columns.extend(gc_features)

# 确保列名唯一且有效
feature_columns = list(dict.fromkeys(feature_columns)) # 移除重复项
valid_features = [f for f in feature_columns if f in df.columns and df[f].dtype in ['int64', 'float64']]

print(f"预处理完成。使用了 {len(valid_features)} 个有效特征。")
print("\n数据集中的异常类别分布:")
print(df['Detailed_Abnormality'].value_counts())
# 准备特征和目标变量
X = df[valid_features].copy()
# 我们的目标是二分类：0 代表 'Normal', 1 代表 'Any_Abnormal'
# 'Is_Abnormal' 列已经完美地满足了这个需求
y_binary = df['Is_Abnormal'].copy()

# 标签名称，用于后续结果解释
label_names = {0: 'Normal', 1: 'Any_Abnormal'}
print("标签映射:")
print(label_names)

# 拆分训练集和测试集（必须在SMOTE之前进行！）
# 使用 stratify 确保训练集和测试集中的异常比例与原始数据一致
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, 
    test_size=0.3, 
    random_state=42, 
    stratify=y_binary
)

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n原始训练集形状: {X_train_scaled.shape}")
print("原始训练集类别分布:")
print(y_train.value_counts())

# 应用SMOTE进行数据增强
# SMOTE只会对训练数据进行过采样，以避免数据泄露
# 对于二分类问题，SMOTE将自动增加少数类（异常样本）的数量
print(f"\n应用SMOTE处理不平衡...")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

print(f"\nSMOTE增强后的训练集形状: {X_train_smote.shape}")
print("SMOTE增强后的训练集类别分布:")
print(y_train_smote.value_counts())

# 初始化并训练随机森林模型
# class_weight='balanced' 会自动为样本量少的类别赋予更高的权重
# 即使使用了SMOTE，保留这个参数也可以进一步增强模型对少数类的关注
rf_model = RandomForestClassifier(
    n_estimators=200,       # 增加树的数量以提高稳定性
    class_weight='balanced',
    random_state=42,
    max_depth=10,           # 限制树的深度以防过拟合
    min_samples_leaf=5      # 限制叶节点最少样本数
)

print("开始在SMOTE增强后的训练集上训练随机森林模型...")
rf_model.fit(X_train_smote, y_train_smote)
print("模型训练完成。")

# 在原始测试集上进行预测
print("\n在原始测试集上进行评估...")
y_pred = rf_model.predict(X_test_scaled)

# 定义目标类别名称
target_names = [label_names[i] for i in sorted(label_names.keys())]

# 打印分类报告
print("\n分类性能报告:")
report = classification_report(y_test, y_pred, target_names=target_names)
print(report)

# 打印总体准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"\n模型总体准确率: {accuracy:.4f}")
# 计算混淆矩阵
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)

# 可视化混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': '样本数量'})
plt.title('随机森林二分类模型混淆矩阵 (SMOTE增强)', fontproperties=font_prop, fontsize=16)
plt.ylabel('真实类别', fontproperties=font_prop, fontsize=12)
plt.xlabel('预测类别', fontproperties=font_prop, fontsize=12)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()

# 确保保存目录存在
os.makedirs('../../Paper/C_4', exist_ok=True)
plt.savefig('../../Paper/C_4/smote_rf_binary_confusion_matrix.png', dpi=300)
plt.show()

# 特征重要性分析
feature_importances = pd.DataFrame({
    '特征': valid_features,
    '重要性': rf_model.feature_importances_
}).sort_values('重要性', ascending=False)

print("\n模型特征重要性 (Top 10):")
display(feature_importances.head(10))

# 可视化特征重要性
plt.figure(figsize=(10, 8))
sns.barplot(x='重要性', y='特征', data=feature_importances.head(15), palette='viridis')
plt.title('特征重要性 Top 15 (二分类模型)', fontproperties=font_prop, fontsize=16)
plt.xlabel('重要性', fontproperties=font_prop, fontsize=12)
plt.ylabel('特征', fontproperties=font_prop, fontsize=12)
plt.tight_layout()
plt.savefig('../../Paper/C_4/smote_rf_binary_feature_importance.png', dpi=300)
plt.show()
# --- 第二阶段：专家规则归因 ---

# 1. 定义从第一个notebook中学到的动态阈值
# 这些阈值是在平衡了精确率和召回率后得到的
dynamic_thresholds = {
    'T13': 2.2, 
    'T18': 1.5, 
    'T21': 1.9
}
print("使用的动态Z值阈值:")
print(dynamic_thresholds)

# 2. 定义归因函数
def attribute_abnormality(z13, z18, z21, thresholds):
    """根据Z值和动态阈值进行异常归因"""
    abnormalities = []
    if abs(z13) > thresholds['T13']:
        abnormalities.append('T13')
    if abs(z18) > thresholds['T18']:
        abnormalities.append('T18')
    if abs(z21) > thresholds['T21']:
        abnormalities.append('T21')
    
    if not abnormalities:
        # 如果AI模型认为是异常，但Z值规则未发现任何异常
        # 这可能是模型捕捉到的、更复杂的非Z值模式，或仅仅是假阳性
        # 我们将其标记为 "Abnormal_Unspecified" (未明确的异常)
        return 'Abnormal_Unspecified'
    
    # 根据T13, T18, T21的组合来确定最终标签
    # 这部分逻辑需要与原始数据中的标签格式保持一致
    if 'T13' in abnormalities and 'T18' in abnormalities:
        return 'T13T18'
    if 'T13' in abnormalities and 'T21' in abnormalities:
        return 'T13T21'
    if 'T18' in abnormalities and 'T21' in abnormalities:
        return 'T18T21'
    if 'T13' in abnormalities:
        return 'T13'
    if 'T18' in abnormalities:
        return 'T18'
    if 'T21' in abnormalities:
        return 'T21'
    
    return 'Abnormal_Unspecified' # 兜底

# --- 应用完整的两阶段流程到测试集 ---

# 第一阶段的预测结果 (0=Normal, 1=Any_Abnormal)
stage1_preds = y_pred

# 获取测试集的原始数据，以便提取Z值
X_test_original = df.loc[X_test.index]

final_predictions = []
for i, prediction in enumerate(stage1_preds):
    if prediction == 0:  # 第一阶段预测为 'Normal'
        final_predictions.append('Normal')
    else:  # 第一阶段预测为 'Any_Abnormal'，进入第二阶段
        sample = X_test_original.iloc[i]
        z13 = sample['13号染色体的Z值']
        z18 = sample['18号染色体的Z值']
        z21 = sample['21号染色体的Z值']
        
        # 应用规则进行归因
        stage2_result = attribute_abnormality(z13, z18, z21, dynamic_thresholds)
        final_predictions.append(stage2_result)

# 将列表转换为Series，便于后续分析
final_predictions = pd.Series(final_predictions, index=X_test.index)

print("\n两阶段流程完成。")
print("最终预测结果分布:")
print(final_predictions.value_counts())
# 获取测试集的真实多分类标签
y_test_true_multiclass = df.loc[X_test.index, 'Detailed_Abnormality']

# 整合所有出现过的类别，以生成完整的报告
all_labels = sorted(list(set(y_test_true_multiclass) | set(final_predictions)))

print("--- 最终两阶段模型性能评估 ---")
print("\n最终分类性能报告:")
final_report = classification_report(
    y_test_true_multiclass, 
    final_predictions, 
    labels=all_labels,
    zero_division=0
)
print(final_report)

# 计算并可视化最终的混淆矩阵
final_cm = confusion_matrix(y_test_true_multiclass, final_predictions, labels=all_labels)
final_cm_df = pd.DataFrame(final_cm, index=all_labels, columns=all_labels)

plt.figure(figsize=(12, 10))
sns.heatmap(final_cm_df, annot=True, fmt='d', cmap='Greens', cbar_kws={'label': '样本数量'})
plt.title('最终两阶段模型混淆矩阵', fontproperties=font_prop, fontsize=16)
plt.ylabel('真实类别', fontproperties=font_prop, fontsize=12)
plt.xlabel('预测类别', fontproperties=font_prop, fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('../../Paper/C_4/final_two_stage_confusion_matrix.png', dpi=300)
plt.show()
# 创建一个DataFrame来对比真实标签和最终预测标签
comparison_df = pd.DataFrame({
    '真实类别': y_test_true_multiclass,
    '预测类别': final_predictions
})

# 计算交叉表，统计每个真实类别被预测为各种类别的数量
crosstab = pd.crosstab(comparison_df['真实类别'], comparison_df['预测类别'])

# 筛选出所有异常类别，不包括'Normal'，并且只选择在测试集真实标签中实际存在的类别
abnormal_labels = [label for label in all_labels if label != 'Normal']
# 只选择在crosstab行索引中实际存在的异常类别
existing_abnormal_labels = [label for label in abnormal_labels if label in crosstab.index]

print(f"所有异常类别: {abnormal_labels}")
print(f"测试集中实际存在的异常类别: {existing_abnormal_labels}")

if existing_abnormal_labels:
    crosstab_abnormal = crosstab.loc[existing_abnormal_labels]
    print("\n真实异常样本的预测分布交叉表:")
    display(crosstab_abnormal)
else:
    print("\n测试集中没有异常样本，无法生成异常样本的预测分布交叉表。")
    crosstab_abnormal = pd.DataFrame()

# 绘制堆叠条形图（只在有数据时绘制）
if not crosstab_abnormal.empty:
    fig, ax = plt.subplots(figsize=(14, 8))
    crosstab_abnormal.plot(kind='bar', stacked=True, ax=ax, 
                           colormap='viridis', width=0.7)

    # 添加标题和标签
    ax.set_title('真实异常样本的最终预测分布', fontproperties=font_prop, fontsize=18, pad=20)
    ax.set_xlabel('真实异常类别', fontproperties=font_prop, fontsize=14, labelpad=15)
    ax.set_ylabel('样本数量', fontproperties=font_prop, fontsize=14, labelpad=15)

    # 美化图表
    ax.tick_params(axis='x', rotation=45, labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(title='预测类别', prop={'size': 12})

    # 在每个堆叠块上添加数值标签
    for c in ax.containers:
        labels = [int(v.get_height()) if v.get_height() > 0 else '' for v in c]
        ax.bar_label(c, labels=labels, label_type='center', color='white', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('../../Paper/C_4/final_prediction_vs_true_distribution.png', dpi=300)
    plt.show()
else:
    print("由于测试集中没有异常样本，跳过异常样本预测分布图的绘制。")
    
y_train_true_multiclass = X_train_original_features['Detailed_Abnormality']

# 初始化训练集的最终预测Series
final_predictions_train = pd.Series(index=X_train.index, dtype=object)

# 对训练集中被AI判定为异常的样本进行归因（修复版）
abnormal_indices_train = X_train.index[y_train_pred_binary == 1]
print(f"   训练集中被AI判定为异常的样本数: {len(abnormal_indices_train)}")

if len(abnormal_indices_train) > 0:
    # 修复：使用lambda函数正确传递参数给attribute_abnormality函数
    attributed_results_train = X_train_original_features.loc[abnormal_indices_train].apply(
        lambda row: attribute_abnormality(
            row['13号染色体的Z值'], 
            row['18号染色体的Z值'], 
            row['21号染色体的Z值'], 
            dynamic_thresholds
        ),
        axis=1
    )
    final_predictions_train.loc[abnormal_indices_train] = attributed_results_train
    print(f"   训练集异常样本归因完成: {len(attributed_results_train)}")
else:
    print("   训练集中没有被AI判定为异常的样本")

# 填充被AI判定为正常的样本
normal_indices_train = X_train.index[y_train_pred_binary == 0]
final_predictions_train.loc[normal_indices_train] = 'Normal'
print(f"   训练集正常样本数: {len(normal_indices_train)}")
print("✅ 训练集预测完成。")

# 2. 对测试集进行预测
print("\n📊 处理测试集...")
X_test_for_pred_scaled = scaler.transform(X_test)
y_test_pred_binary = rf_model.predict(X_test_for_pred_scaled)

# 获取测试集的原始特征
X_test_original_features = df.loc[X_test.index]

# 初始化测试集的最终预测Series
final_predictions_test = pd.Series(index=X_test.index, dtype=object)

# 对测试集中被AI判定为异常的样本进行归因（修复版）
abnormal_indices_test = X_test.index[y_test_pred_binary == 1]
print(f"   测试集中被AI判定为异常的样本数: {len(abnormal_indices_test)}")

if len(abnormal_indices_test) > 0:
    # 修复：使用lambda函数正确传递参数给attribute_abnormality函数
    attributed_results_test = X_test_original_features.loc[abnormal_indices_test].apply(
        lambda row: attribute_abnormality(
            row['13号染色体的Z值'], 
            row['18号染色体的Z值'], 
            row['21号染色体的Z值'], 
            dynamic_thresholds
        ),
        axis=1
    )
    final_predictions_test.loc[abnormal_indices_test] = attributed_results_test
    print(f"   测试集异常样本归因完成: {len(attributed_results_test)}")
else:
    print("   测试集中没有被AI判定为异常的样本")

# 填充被AI判定为正常的样本
normal_indices_test = X_test.index[y_test_pred_binary == 0]
final_predictions_test.loc[normal_indices_test] = 'Normal'
print(f"   测试集正常样本数: {len(normal_indices_test)}")
print("✅ 测试集预测完成。")

# 3. 合并所有预测结果
print("\n📊 合并预测结果...")
final_predictions_all = pd.concat([final_predictions_train, final_predictions_test])
print(f"总预测样本数: {len(final_predictions_all)}")

# 4. 显示预测结果统计
print("\n📈 预测结果统计:")
prediction_counts = final_predictions_all.value_counts().sort_index()
for category, count in prediction_counts.items():
    percentage = count / len(final_predictions_all) * 100
    print(f"   {category}: {count} ({percentage:.2f}%)")

print("\n✅ 全数据集预测完成！")
print("="*70)

# --- 可视化全数据集上的最终预测结果分布 ---
print("正在可视化模型在全数据集上的预测结果分布...")

# 1. 计算预测结果的分布
prediction_counts = final_predictions_all.value_counts()

# 2. 绘制条形图
fig, ax = plt.subplots(figsize=(12, 7))
prediction_counts.plot(kind='bar', ax=ax, color='skyblue', width=0.7)

# 添加标题和标签
ax.set_title('模型在全数据集上的最终预测结果分布', fontproperties=font_prop, fontsize=18, pad=20)
ax.set_xlabel('预测类别', fontproperties=font_prop, fontsize=14, labelpad=15)
ax.set_ylabel('样本数量', fontproperties=font_prop, fontsize=14, labelpad=15)

# 美化图表
ax.tick_params(axis='x', rotation=45, labelsize=12)
ax.grid(axis='y', linestyle='--', alpha=0.6)

# 在条形图顶部添加数值标签
for i, count in enumerate(prediction_counts):
    ax.text(i, count + 3, str(count), ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_ylim(0, prediction_counts.max() * 1.15) # 调整y轴范围以容纳标签

plt.tight_layout()
plt.savefig('../../Paper/C_4/full_dataset_prediction_distribution.png', dpi=300)
plt.show()

print("图表已保存至 ../../Paper/C_4/full_dataset_prediction_distribution.png")

## 8. 真实分布 vs 预测分布对比可视化

# 获取全数据集的真实标签分布
true_distribution = df['Detailed_Abnormality'].value_counts()
pred_distribution = final_predictions_all.value_counts()

print("\n=== 真实分布 vs 预测分布对比 ===")
print("\n真实标签分布:")
print(true_distribution)
print("\n预测标签分布:")
print(pred_distribution)

# 合并所有可能出现的类别
all_categories = sorted(list(set(true_distribution.index) | set(pred_distribution.index)))

# 确保两个分布都包含所有类别（用0填充缺失的类别）
true_counts = [true_distribution.get(cat, 0) for cat in all_categories]
pred_counts = [pred_distribution.get(cat, 0) for cat in all_categories]

# 创建对比条形图
x = np.arange(len(all_categories))
width = 0.35

fig, ax = plt.subplots(figsize=(15, 8))

# 绘制两组条形图
bars1 = ax.bar(x - width/2, true_counts, width, label='真实分布', color='lightcoral', alpha=0.8)
bars2 = ax.bar(x + width/2, pred_counts, width, label='预测分布', color='skyblue', alpha=0.8)

# 添加标题和标签
ax.set_title('全数据集：真实分布 vs 模型预测分布对比', fontproperties=font_prop, fontsize=18, pad=20)
ax.set_xlabel('类别', fontproperties=font_prop, fontsize=14, labelpad=15)
ax.set_ylabel('样本数量', fontproperties=font_prop, fontsize=14, labelpad=15)
ax.set_xticks(x)
ax.set_xticklabels(all_categories, rotation=45, ha='right')
ax.legend(prop={'size': 12})

# 添加网格
ax.grid(axis='y', linestyle='--', alpha=0.6)

# 在条形图上添加数值标签
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        if height > 0:  # 只在高度大于0时添加标签
            ax.annotate(f'{int(height)}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=10, fontweight='bold')

add_value_labels(bars1)
add_value_labels(bars2)

# 调整y轴范围
max_count = max(max(true_counts), max(pred_counts))
ax.set_ylim(0, max_count * 1.15)

plt.tight_layout()
plt.savefig('../../Paper/C_4/true_vs_predicted_distribution_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n对比图表已保存至 ../../Paper/C_4/true_vs_predicted_distribution_comparison.png")

## 9. 详细性能分析表格

# 计算混淆矩阵并生成详细分析
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, matthews_corrcoef, balanced_accuracy_score
import pandas as pd

# 生成分类报告
y_true_all = df['Detailed_Abnormality']
classification_rep = classification_report(y_true_all, final_predictions_all, output_dict=True, zero_division=0)

# 转换为DataFrame以便更好地显示
metrics_df = pd.DataFrame(classification_rep).transpose()
# 只保留我们关心的指标，并重新排序
metrics_df = metrics_df[['precision', 'recall', 'f1-score', 'support']].round(4)

print("\n=== 各类别详细性能指标 ===")
display(metrics_df)

# 计算混淆矩阵
all_categories = sorted(list(set(y_true_all) | set(final_predictions_all)))
cm_full = confusion_matrix(y_true_all, final_predictions_all, labels=all_categories)
cm_df = pd.DataFrame(cm_full, index=all_categories, columns=all_categories)

print("\n=== 完整混淆矩阵 ===")
display(cm_df)

# 可视化混淆矩阵
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 10))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': '样本数量'})
plt.title('全数据集混淆矩阵 - 真实 vs 预测', fontproperties=font_prop, fontsize=16)
plt.ylabel('真实类别', fontproperties=font_prop, fontsize=12)
plt.xlabel('预测类别', fontproperties=font_prop, fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('../../Paper/C_4/full_dataset_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n混淆矩阵图表已保存至 ../../Paper/C_4/full_dataset_confusion_matrix.png")

# --- 关键性能指标汇总 ---

# 计算二分类性能（Normal vs Any_Abnormal）
y_true_binary = (y_true_all != 'Normal').astype(int)
y_pred_binary = (final_predictions_all != 'Normal').astype(int)
from sklearn.metrics import precision_recall_fscore_support
binary_precision, binary_recall, binary_f1, _ = precision_recall_fscore_support(
    y_true_binary, y_pred_binary, average='binary'
)

# 计算更丰富的评估指标
kappa = cohen_kappa_score(y_true_all, final_predictions_all)
balanced_acc = balanced_accuracy_score(y_true_all, final_predictions_all)
mcc = matthews_corrcoef(y_true_binary, y_pred_binary)


print(f"\n" + "="*15 + " 模型综合性能评估 " + "="*15)

print(f"\n--- 总体性能 ---")
print(f"总体准确率 (Overall Accuracy): {classification_rep['accuracy']:.4f}")
print(f"平衡准确率 (Balanced Accuracy): {balanced_acc:.4f}")
print(f"宏平均F1分数 (Macro Avg F1): {classification_rep['macro avg']['f1-score']:.4f}")
print(f"加权平均F1分数 (Weighted Avg F1): {classification_rep['weighted avg']['f1-score']:.4f}")
print(f"科恩系数 (Cohen's Kappa): {kappa:.4f}")

print(f"\n--- 二分类性能 (正常 vs. 异常) ---")
print(f"异常检测精确率 (Precision): {binary_precision:.4f}")
print(f"异常检测召回率 (Recall/Sensitivity): {binary_recall:.4f}")
print(f"异常检测F1分数 (F1-Score): {binary_f1:.4f}")
print(f"马修斯相关系数 (MCC): {mcc:.4f}")


print("\n" + "="*17 + " 指标解读 " + "="*17)
print("平衡准确率 (Balanced Accuracy): 在不平衡数据中比标准准确率更具参考价值，它对每个类别的召回率进行平均。")
print("科恩系数 (Cohen's Kappa): 衡量分类结果与随机分类相比的提升程度，值域为[-1, 1]，0表示与随机分类无异，1表示完美分类。")
print("马修斯相关系数 (MCC): 评估二分类性能的均衡指标，综合考虑了四项混淆矩阵元素，即使在类别极不平衡时也表现稳健，值域为[-1, 1]，是反映模型整体性的重要指标。")