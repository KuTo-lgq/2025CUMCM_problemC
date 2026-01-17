import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy.stats import norm
import warnings

# --- 前置设定 ---
warnings.filterwarnings("ignore")

# --- 任务1: 加载数据与BMI分组 (与Q2一致) ---
try:
    df_processed = pd.read_csv('E:\\Mathematics_Modeling_study\\2025_CUMCM\\Code\\C_1\\processed_nipt_data.csv')
except FileNotFoundError:
    print("错误：请确保 'processed_nipt_data.csv' 文件与本代码在同一目录下。")
    exit()

df_male = df_processed[df_processed['Y染色体浓度'].notna()].copy()
df_male['Y染色体浓度(%)'] = df_male['Y染色体浓度'] * 100

bins = [25, 30, 35, 40, np.inf]
labels = ['超重组 (25-29.9)', 'I级肥胖组 (30-34.9)', 'II级肥胖组 (35-39.9)', 'III级肥胖组 (≥40)']
df_male['BMI分组'] = pd.cut(df_male['孕妇BMI'], bins=bins, labels=labels, right=False)
df_grouped = df_male.dropna(subset=['BMI分组']).copy()

print("--- 任务1: 各BMI分组的样本数量 ---")
print(df_grouped['BMI分组'].value_counts().sort_index())
print("-" * 45)


# --- 任务2: 构建增强版预测模型 (综合多因素) ---
df_grouped.rename(columns={
    'Y染色体浓度(%)': 'c_y',
    '孕周数': 'gest_week',
    '孕妇代码': 'patient_code',
    '年龄': 'age',
    '身高': 'height',
    '体重': 'weight'
}, inplace=True)

models_q3 = {}
group_means = {} # 用于存储每个组的均值，以便后续预测

print("\n--- 任务2: 为各分组建立增强版动态预测模型 ---")
for group_name in labels:
    print(f"正在为【{group_name}】建立模型...")
    group_data = df_grouped[df_grouped['BMI分组'] == group_name]
    
    # 存储该组的平均协变量，用于后续预测
    group_means[group_name] = {
        'age': group_data['age'].mean(),
        'height': group_data['height'].mean(),
        'weight': group_data['weight'].mean()
    }
    
    # 定义包含多个协变量的增强版模型公式
    model_formula_q3 = "c_y ~ gest_week + age + height + weight"
    
    try:
        model = smf.mixedlm(
            model_formula_q3, 
            data=group_data, 
            groups=group_data["patient_code"],
            re_formula="~gest_week"
        ).fit(method='powell')
        
        models_q3[group_name] = model
        print(f"【{group_name}】模型建立完成。")
    except Exception as e:
        print(f"为【{group_name}】建立模型时出错: {e}")

print("-" * 45)


# --- 任务3: 使用增强模型求解最佳NIPT时点 ---
print("\n--- 任务3: 基于总成本最小化求解最优NIPT时点 (集成风险成本框架) ---")

# 定义延迟风险成本函数 (与问题二完全一致)
def cost_delay_function(T):
    """根据孕周计算延迟风险成本"""
    if T <= 12:
        return T
    elif T <= 27:
        return 12 + 2.5 * (T - 12)
    else:
        return 49.5 + 5 * (T - 27)

results_q3 = []
FAILURE_PENALTY_COEFFICIENT = 100  # 失败惩罚系数 C (与问题二一致)
threshold = 4.0 
weeks_to_check = np.arange(10.0, 25.1, 0.1) 

for group_name, model in models_q3.items():
    
    min_total_cost = np.inf
    optimal_result = {}
    
    means = group_means[group_name]
    cov_re = model.cov_re 
    resid_var = model.scale

    for week in weeks_to_check:
        # 使用组内平均值构建预测数据点
        pred_data = pd.DataFrame({
            'gest_week': [week],
            'age': [means['age']],
            'height': [means['height']],
            'weight': [means['weight']]
        })
        
        mean_pred = model.predict(pred_data)[0]
        
        # 计算预测方差
        X = np.array([1, week, means['age'], means['height'], means['weight']]) # 新的设计矩阵
        cov_fe = model.cov_params().iloc[:len(X), :len(X)]
        var_fe = X @ cov_fe @ X.T
        var_re = cov_re.iloc[0,0] + (week**2 * cov_re.iloc[1,1]) + (2 * week * cov_re.iloc[0,1])
        total_var = var_fe + var_re + resid_var
        total_std = np.sqrt(total_var)
        
        prob_success = norm.sf(threshold, loc=mean_pred, scale=total_std)
        prob_failure = 1 - prob_success

        # 计算总成本 (与问题二框架一致)
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
            
    results_q3.append(optimal_result)

# --- 任务4: 结果汇总与分析 ---
df_results_q3 = pd.DataFrame(results_q3)
print("\n--- 最终结果：基于总成本最小化的各BMI分组最佳NIPT时点推荐 (多因素增强版) ---")
print(df_results_q3.to_string(index=False))
print("-" * 45)

print("\n--- 分析总结 ---")
print("通过集成延迟风险成本和检测失败成本的综合优化框架:")
print("1. 模型不再寻找固定成功率的最早时间点，而是寻找总成本最小的最优平衡点")
print("2. 每个BMI分组都得到了唯一的最佳推荐时点，实现了个性化精准医疗建议")
print("3. 成本明细显示了模型在延迟风险和检测失败风险之间的智能权衡")