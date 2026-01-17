import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy.stats import norm
import warnings

# --- 前置设定 ---
# 忽略一些在模型拟合过程中可能出现的警告，不影响最终结果
warnings.filterwarnings("ignore")

# --- 任务1: 加载您已处理好的数据并进行BMI分组 ---

# 从您提供的已处理好的文件中加载数据
try:
    df_processed = pd.read_csv('E:\\Mathematics_Modeling_study\\2025_CUMCM\\Code\\C_1\\processed_nipt_data.csv')
except FileNotFoundError:
    print("错误：请确保 'processed_nipt_data.csv' 文件存在于正确的路径。")
    exit()

# 筛选出男胎数据用于本问分析 (Y染色体浓度列存在即为男胎)
df_male = df_processed[df_processed['Y染色体浓度'].notna()].copy()
df_male['Y染色体浓度(%)'] = df_male['Y染色体浓度'] * 100


# 定义分组的边界和标签 (根据临床标准)
bins = [25, 30, 35, 40, np.inf]
labels = ['超重组 (25-29.9)', 'I级肥胖组 (30-34.9)', 'II级肥胖组 (35-39.9)', 'III级肥胖组 (≥40)']

# 创建新的BMI分组列
df_male['BMI分组'] = pd.cut(df_male['孕妇BMI'], bins=bins, labels=labels, right=False)

# 移除不在这些分组内的样本 (例如BMI < 25) 并查看各组样本量
df_grouped = df_male.dropna(subset=['BMI分组']).copy()

print("--- 任务1: 各BMI分组的样本数量 ---")
print(df_grouped['BMI分组'].value_counts().sort_index())
print("-" * 45)


# --- 任务2: 为每个BMI分组建立独立的动态预测模型 ---

df_grouped.rename(columns={
    'Y染色体浓度(%)': 'c_y',
    '孕周数': 'gest_week',
    '孕妇代码': 'patient_code',
    '年龄': 'age'
}, inplace=True)


models = {}
print("\n--- 任务2: 为各分组建立动态预测模型 ---")
for group_name in labels:
    print(f"正在为【{group_name}】建立模型...")
    group_data = df_grouped[df_grouped['BMI分组'] == group_name]
    
    try:
        model = smf.mixedlm(
            "c_y ~ gest_week", 
            data=group_data, 
            groups=group_data["patient_code"],
            re_formula="~gest_week"
        ).fit(method='powell')
        
        models[group_name] = model
        print(f"【{group_name}】模型建立完成。")
    except Exception as e:
        print(f"为【{group_name}】建立模型时出错: {e}")

print("-" * 45)


# --- 任务3 & 4 (重构): 基于总成本最小化求解最优NIPT时点 ---
print("\n--- 任务3 & 4: 基于总成本最小化求解最优NIPT时点 ---")

# 1. 定义延迟风险成本函数
def cost_delay_function(T):
    """根据孕周计算延迟风险成本"""
    if T <= 12:
        return T
    elif T <= 27:
        return 12 + 2.5 * (T - 12)
    else:
        return 49.5 + 5 * (T - 27)

# 2. 定义参数
FAILURE_PENALTY_COEFFICIENT = 100  # 失败惩罚系数 C
threshold = 4.0
weeks_to_check = np.arange(10.0, 25.1, 0.1)

results = []

for group_name, model in models.items():
    
    min_total_cost = np.inf
    optimal_result = {}
    
    cov_re = model.cov_re 
    resid_var = model.scale

    for week in weeks_to_check:
        # 计算成功概率
        pred_data = pd.DataFrame({'gest_week': [week]})
        mean_pred = model.predict(pred_data)[0]
        
        X = np.array([1, week])
        cov_fe = model.cov_params().iloc[:2, :2]
        var_fe = X @ cov_fe @ X.T
        var_re = cov_re.iloc[0,0] + (week**2 * cov_re.iloc[1,1]) + (2 * week * cov_re.iloc[0,1])
        total_var = var_fe + var_re + resid_var
        total_std = np.sqrt(total_var)
        
        prob_success = norm.sf(threshold, loc=mean_pred, scale=total_std)
        prob_failure = 1 - prob_success

        # 计算总成本
        cost_delay = cost_delay_function(week)
        cost_failure = FAILURE_PENALTY_COEFFICIENT * prob_failure
        total_cost = cost_delay + cost_failure

        # 寻找成本最低的点
        if total_cost < min_total_cost:
            min_total_cost = total_cost
            optimal_result = {
                'BMI分组': group_name,
                '推荐最佳时点': f"{week:.1f}周",
                '此时点成功率': f"{prob_success:.1%}",
                '延迟风险成本': f"{cost_delay:.1f}",
                '检测失败成本': f"{cost_failure:.1f}",
                '总成本': f"{total_cost:.1f}"
            }
            
    results.append(optimal_result)

# --- 结果汇总与展示 ---
df_results = pd.DataFrame(results)
print("\n--- 最终结果：各BMI分组的最佳NIPT时点推荐 (基于总成本最小化) ---")
print(df_results.to_string(index=False))
print("-" * 45)