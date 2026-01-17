import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import statsmodels.api as sm
import random

# --- 图表样式与中文支持设置 ---
# 设置绘图样式
plt.style.use('seaborn-v0_8-whitegrid')
# 设置字体以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False

try:
    # --- 1. 加载数据并重新拟合模型 ---
    print("步骤 1: 加载数据并重新拟合模型...")
    df = pd.read_csv('processed_nipt_data.csv')

    # --- 数据准备：与建模时保持一致 ---
    # 1. Y染色体浓度单位转换 (小数 -> 百分比)
    # 这是确保模型评估与建模使用相同数据尺度的关键步骤
    df['Y染色体浓度'] = df['Y染色体浓度'] * 100
    print("Y染色体浓度已转换为百分比。")
    
    # 2. 将关键列重命名为ASCII字符，以便在公式中使用
    rename_dict = {
        'Y染色体浓度': 'y_concentration',
        '孕周数': 'gestational_week',  # 修正：CSV中的列名为'孕周数'
        '孕妇BMI': 'bmi',
        '孕妇代码': 'patient_id'
    }
    df.rename(columns=rename_dict, inplace=True)
    df.dropna(subset=['gestational_week'], inplace=True)
    print("列名已重命名。")

    # 定义并拟合线性混合效应模型
    model_formula = "y_concentration ~ gestational_week + bmi"
    model = smf.mixedlm(model_formula, df, groups=df['patient_id'], re_formula="~gestational_week")
    result = model.fit()
    print("模型重新拟合成功。")

    # --- 2. 残差分析 ---
    print("\n步骤 2: 进行残差分析...")
    # 获取模型的残差和拟合值
    residuals = result.resid
    fitted_values = result.fittedvalues

    # 图 1: 残差 vs. 拟合值图
    plt.figure(figsize=(10, 6))
    plt.scatter(fitted_values, residuals, alpha=0.5, color='skyblue', edgecolor='k')
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Fitted Values (拟合值)')
    plt.ylabel('Residuals (残差)')
    plt.title('Residuals vs. Fitted Values Plot (残差与拟合值图)')
    plt.savefig('residuals_vs_fitted.png')
    plt.show()
    print("已生成图片 'residuals_vs_fitted.png'")

    # 图 2: 残差的正态性Q-Q图
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    sm.qqplot(residuals, line='45', fit=True, ax=ax)
    ax.set_title('Q-Q Plot of Residuals (残差Q-Q图)')
    plt.savefig('qq_plot.png')
    plt.show()
    print("已生成图片 'qq_plot.png'")
    plt.close(fig) # 关闭由qqplot创建的图形

    # --- 3. 可视化预测检验 ---
    print("\n步骤 3: 进行可视化预测检验...")
    # 计算每个数据点的预测值并添加到DataFrame中
    df['predicted'] = result.predict(df)

    # 从所有孕妇中随机选择6位进行可视化
    unique_patients = df['patient_id'].unique()
    sample_patients = random.sample(list(unique_patients), min(len(unique_patients), 6))
    
    # 创建子图网格
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
    axes = axes.flatten()

    # 遍历选中的孕妇并绘图
    for i, patient_id in enumerate(sample_patients):
        ax = axes[i]
        patient_data = df[df['patient_id'] == patient_id].sort_values('gestational_week')
        
        # 绘制真实观测值（散点图）
        ax.scatter(patient_data['gestational_week'], patient_data['y_concentration'], label='Actual Data (真实值)', zorder=5, s=60)
        
        # 绘制模型预测值（连线图）
        ax.plot(patient_data['gestational_week'], patient_data['predicted'], color='red', linestyle='-', marker='o', label='Model Prediction (模型预测)')
        
        ax.set_title(f'Patient ID (孕妇代码): {patient_id}')
        ax.set_xlabel('Gestational Week (孕周)')
        ax.set_ylabel('Y-chromosome Concentration (%)')
        ax.legend()

    # 隐藏多余的子图
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Visual Predictive Check for Sample Patients (样本孕妇可视化预测检验)', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('visual_predictive_check.png')
    plt.show()
    print("已生成图片 'visual_predictive_check.png'")
    print("\n模型验证过程已全部完成。")

except FileNotFoundError:
    print("错误：文件 'processed_nipt_data.csv' 未找到。请确保该文件与脚本在同一目录下。")
except KeyError as e:
    print(f"程序运行出错 (KeyError): {e}。请检查CSV文件中的列名是否与代码中的预期一致。")
except Exception as e:
    print(f"程序运行出错: {e}")