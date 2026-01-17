import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
# 读取数据
data = pd.read_excel('E:\\Mathematics_Modeling_study\\2025_CUMCM\\Stem\\C题\\附件.xlsx')
# 数据预处理函数
def preprocess_data(data, is_male=True):
    def convert_gestational_age(ga):
        if pd.isna(ga) or not isinstance(ga, str):
            return np.nan
        # 解析 "周w+天数" 格式，如 "10w+3" 转换为 10.43 周
        match = re.match(r'(\d+)[wW](?:\+(\d+)(?:d)?)?', str(ga).strip(), re.IGNORECASE)
        if match:
            weeks = int(match.group(1))
            days = int(match.group(2)) if match.group(2) else 0
            return weeks + days / 7
        print(f"警告：无法解析孕周格式 '{ga}'，返回 NaN")
        return np.nan

    data = data.copy()
    data['孕周数'] = data['检测孕周'].apply(convert_gestational_age)

    # 移除不必要的日期计算逻辑
    # try:
    #     data['末次月经'] = pd.to_datetime(data['末次月经'], errors='coerce')
    #     data['检测日期'] = pd.to_datetime(data['检测日期'], format='%Y%m%d', errors='coerce')
    #     data['孕周数计算'] = (data['检测日期'] - data['末次月经']).dt.days / 7
    # except Exception as e:
    #     print(f"日期转换错误: {e}")
    #     data['孕周数计算'] = np.nan

    data['BMI计算'] = data['体重'] / ((data['身高'] / 100) ** 2)

    if is_male and 'Y染色体浓度' in data.columns:
        data = data[(data['Y染色体浓度'] >= 0) & (data['Y染色体浓度'] <= 1)]
        # 筛选孕周 10~25 周且 Y 染色体浓度 >= 4% 的男胎数据
        data = data[(data['孕周数'] >= 10) & (data['孕周数'] <= 25) & (data['Y染色体浓度'] >= 0.04)]
    elif is_male:
        raise ValueError("数据中缺少 'Y染色体浓度' 列")

    required_columns = ['孕周数', '孕妇BMI', 'Y染色体浓度'] if is_male else ['孕周数', '孕妇BMI']
    data = data.dropna(subset=required_columns)

    if len(data) < 2:
        raise ValueError(f"清洗后数据不足（{len(data)} 行），需要至少 2 行有效数据")

    print(f"筛选后男胎数据行数 (孕周 10-25 周, Y 浓度 >= 4%): {len(data)}")
    return data
# 调用预处理函数
try:
    preprocessed_male_data = preprocess_data(data, is_male=True)
    
    # 显示处理后数据的前几行和基本信息
    print("\n预处理后的数据信息:")
    preprocessed_male_data.info()
    
    print("\n预处理后的数据预览:")
    display(preprocessed_male_data.head())
    
    # 保存处理后的数据到 CSV 文件
    output_path = 'processed_nipt_data.csv'
    preprocessed_male_data.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n处理后的数据已保存到 '{output_path}'")

except ValueError as e:
    print(f"数据预处理失败: {e}")
# 1. Y染色体浓度单位转换 (小数 -> 百分比)
print(f"转换前Y染色体浓度范围: {preprocessed_male_data['Y染色体浓度'].min():.3f} ~ {preprocessed_male_data['Y染色体浓度'].max():.3f}")
preprocessed_male_data['Y染色体浓度'] = preprocessed_male_data['Y染色体浓度'] * 100
print(f"转换后Y染色体浓度范围: {preprocessed_male_data['Y染色体浓度'].min():.1f}% ~ {preprocessed_male_data['Y染色体浓度'].max():.1f}%")
print("✓ Y染色体浓度已转换为百分比单位")

# 2. 重命名孕周列
if '孕周数' in preprocessed_male_data.columns:
    preprocessed_male_data.rename(columns={'孕周数': 'T_weeks'}, inplace=True)
    print("✓ '孕周数' 列已重命名为 'T_weeks'")

# 显示修改后的数据预览
display(preprocessed_male_data[['T_weeks', 'Y染色体浓度', '孕妇BMI']].head())
# 对IVF妊娠进行编码 (自然受孕=0, 辅助生殖技术=1)
preprocessed_male_data['IVF_encoded'] = preprocessed_male_data['IVF妊娠'].map({
    '自然受孕': 0,
    'IVF': 1,
    'IUI（人工授精）': 1,
    'IVF（试管婴儿）': 1
}).fillna(0) # 假设缺失值为自然受孕

# 处理怀孕次数列中的"≥3"字符串
def convert_pregnancy_count(count):
    if pd.isna(count):
        return np.nan
    count_str = str(count).strip()
    if '≥3' in count_str or '>=3' in count_str:
        return 3
    else:
        try:
            return int(count_str)
        except:
            return np.nan
preprocessed_male_data['怀孕次数_encoded'] = preprocessed_male_data['怀孕次数'].apply(convert_pregnancy_count)

# -- BMI分组标准更新 --
# 根据更适合亚洲孕妇的标准对BMI进行分类
def categorize_bmi_asian_pregnancy(bmi):
    if pd.isna(bmi):
        return 'Unknown'
    elif bmi < 18.5:
        return '体重过低'
    elif 18.5 <= bmi < 24.0:
        return '正常体重'
    elif 24.0 <= bmi < 28.0:
        return '超重'
    else: # bmi >= 28.0
        return '肥胖'

preprocessed_male_data['BMI_category'] = preprocessed_male_data['孕妇BMI'].apply(categorize_bmi_asian_pregnancy)

print("✓ 特征工程完成，已更新BMI分组标准。")
print("\n采用新标准后的BMI分组分布情况:")
print(preprocessed_male_data['BMI_category'].value_counts())
import seaborn as sns
import matplotlib.font_manager as fm

# --- 中文字体设置 ---
# (代码从 data_preprocessing.ipynb 借鉴，确保图表中文正常显示)
try:
    font_name = 'Microsoft YaHei' # 尝试使用常见的 Windows 字体
    fm.FontProperties(family=font_name)
    print(f"✓ 找到并使用系统字体: '{font_name}'")
except RuntimeError:
    try:
        font_name = 'SimHei'
        fm.FontProperties(family=font_name)
        print(f"✓ 找到并使用系统字体: '{font_name}'")
    except RuntimeError:
        print("⚠ 警告: 未能找到 'Microsoft YaHei' 或 'SimHei' 字体，中文可能无法显示。")
        font_name = None
sns.set_theme(style="whitegrid", font=font_name)
plt.rcParams['axes.unicode_minus'] = False
# --- 字体设置结束 ---


# 图 1: Y染色体浓度与孕周的关系（按BMI分组着色）
plt.figure(figsize=(10, 6))
scatter = plt.scatter(preprocessed_male_data['T_weeks'], preprocessed_male_data['Y染色体浓度'], 
                      c=preprocessed_male_data['孕妇BMI'], cmap='viridis', alpha=0.6, s=50)
plt.xlabel('孕周 (T_weeks)', fontsize=12)
plt.ylabel('Y染色体浓度 (%)', fontsize=12)
plt.title('Y染色体浓度与孕周的关系', fontsize=14, fontweight='bold')
cbar = plt.colorbar(scatter)
cbar.set_label('BMI (kg/m²)', fontsize=12)
plt.axhline(y=4, color='red', linestyle='--', linewidth=2, alpha=0.8, label='4%检测阈值')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 图 2: BMI、孕周与年龄的分布概览
fig, axes = plt.subplots(1, 3, figsize=(24, 6)) # 修改：扩展为 1x3 子图
# BMI分布
axes[0].hist(preprocessed_male_data['孕妇BMI'].dropna(), bins=30, alpha=0.7, edgecolor='black', color='skyblue')
axes[0].set_xlabel('BMI (kg/m²)', fontsize=12)
axes[0].set_ylabel('频数', fontsize=12)
axes[0].set_title('孕妇BMI分布', fontsize=14, fontweight='bold')
axes[0].axvline(x=24, color='orange', linestyle='--', linewidth=2, label='超重阈值 (24.0)')
axes[0].axvline(x=28, color='red', linestyle='--', linewidth=2, label='肥胖阈值 (28.0)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
# 孕周分布
axes[1].hist(preprocessed_male_data['T_weeks'].dropna(), bins=30, alpha=0.7, edgecolor='black', color='lightgreen')
axes[1].set_xlabel('孕周 (T_weeks)', fontsize=12)
axes[1].set_ylabel('频数', fontsize=12)
axes[1].set_title('检测孕周分布', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)
# 新增：年龄分布
axes[2].hist(preprocessed_male_data['年龄'].dropna(), bins=20, alpha=0.7, edgecolor='black', color='salmon')
axes[2].set_xlabel('年龄 (岁)', fontsize=12)
axes[2].set_ylabel('频数', fontsize=12)
axes[2].set_title('孕妇年龄分布', fontsize=14, fontweight='bold')
axes[2].grid(True, alpha=0.3)
plt.suptitle('数据分布概览', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


# 图 3: 不同BMI组的Y染色体浓度随孕周变化趋势
sns.lmplot(data=preprocessed_male_data, x='T_weeks', y='Y染色体浓度', hue='BMI_category', 
           aspect=1.5, height=6, scatter_kws={'alpha':0.4})
plt.axhline(y=4, color='red', linestyle='-', linewidth=2, alpha=0.8, label='4%检测阈值')
plt.xlabel('孕周 (T_weeks)', fontsize=12)
plt.ylabel('Y染色体浓度 (%)', fontsize=12)
plt.title('不同BMI组的Y染色体浓度随孕周变化趋势', fontsize=14, fontweight='bold')
plt.legend(title='BMI 分组')
plt.grid(True, alpha=0.3)
plt.show()


# 图 4: 扩展后的关键变量相关性分析
corr_vars = [
    '年龄', 'T_weeks', '孕妇BMI', 'Y染色体浓度', 
    'GC含量', '原始读段数', 'X染色体浓度', '21号染色体的Z值'
]
corr_matrix = preprocessed_male_data[corr_vars].corr()
# 增大画布尺寸以便清晰展示
plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
            square=True, cbar_kws={"shrink": .8}, fmt='.2f')
plt.title('关键变量相关性矩阵', fontsize=16, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# 输出与Y染色体浓度相关的重要系数
print("=== 与Y染色体浓度相关的重要系数 ===")
print(corr_matrix['Y染色体浓度'].sort_values(ascending=False))

import statsmodels.formula.api as smf

# 准备用于建模的数据副本
model_data = preprocessed_male_data.copy()

# 为了避免在公式中出现编码问题或特殊字符，将关键列重命名为ASCII字符
rename_dict = {
    'Y染色体浓度': 'y_concentration',
    'T_weeks': 'gestational_week',
    '孕妇BMI': 'bmi',
    '孕妇代码': 'patient_id'
}
model_data.rename(columns=rename_dict, inplace=True)

# 确保关键列没有缺失值
model_data.dropna(subset=['gestational_week', 'bmi', 'y_concentration', 'patient_id'], inplace=True)

# 定义并拟合线性混合效应模型
# 公式表示：Y染色体浓度同时受孕周和BMI的线性影响
# groups=...: 指定了数据的分组结构，即数据点是按孕妇ID嵌套的
# re_formula="~gestational_week": 允许每个孕妇有自己独特的截距(基础水平)和孕周斜率(增长速率)
try:
    model_formula = "y_concentration ~ gestational_week + bmi"
    model = smf.mixedlm(model_formula, model_data, groups=model_data['patient_id'], re_formula="~gestational_week")
    result = model.fit()

    # 打印模型结果的详细摘要
    print(result.summary())

except Exception as e:
    print(f"模型拟合失败: {e}")
    print("请检查数据是否存在问题，例如完美的共线性或数据量过少。")
