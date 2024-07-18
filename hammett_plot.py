import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import json

def generate_hammett_plot(substituents, values, y_axis_label, log_transform, upload_folder):
    # 讀取data.xlsx
    data_file = 'Table_1.xlsx'
    df_data = pd.read_excel(data_file)
    sigma_values = df_data.set_index('substituent')['σp'].to_dict()

    def plot_hammett(x_data, y_data, subs, title, xlabel, ylabel, slope, intercept, r_squared):
        plt.figure(figsize=(10, 6))
        plt.scatter(x_data, y_data)
        for i, txt in enumerate(subs):
            plt.annotate(txt, (x_data[i], y_data[i]))
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        # 繪製回歧線
        plt.plot(x_data, slope * np.array(x_data) + intercept, color='red')

        # 顯示回歧線方程式及R²值
        equation_text = f'y = {slope:.2f}x + {intercept:.2f}\n$R^2$ = {r_squared:.4f}'
        plt.text(0.05, 0.85, equation_text, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

        plt.grid(True)
        plot_image_path = os.path.join(upload_folder, 'hammett_plot.png')
        plt.savefig(plot_image_path)
        plt.close()

        return plot_image_path

    # 準備數據
    sigma = [sigma_values[sub] for sub in substituents]
    
    # 轉換數據類型
    y_data = pd.to_numeric(values, errors='coerce').tolist()
    if log_transform:
        y_data = np.log10(y_data).tolist()
    
    y_axis_label = f'log({y_axis_label})'

    # 確保sigma也是浮點數類型
    sigma = pd.to_numeric(sigma, errors='coerce').tolist()

    # 線性回歸
    X = np.array(sigma).reshape(-1, 1)
    y = np.array(y_data)
    reg = LinearRegression().fit(X, y)
    slope = reg.coef_[0]
    intercept = reg.intercept_
    r_squared = r2_score(y, reg.predict(X))

    plot_image_path = plot_hammett(sigma, y_data, substituents, 'Hammett Plot', 'σ', y_axis_label, slope, intercept, r_squared)

    # 準備數據以JSON格式存儲
    data_json = {
        "substituent": substituents,
        y_axis_label: values,
        "σ": sigma,
        "ρ": slope,
        "R²": r_squared
    }

    # 生成output.xlsx
    output_path = os.path.join(upload_folder, 'output.xlsx')
    df_output = pd.DataFrame({'data': [json.dumps(data_json, ensure_ascii=False)]})
    df_output.to_excel(output_path, index=False)

    return plot_image_path, output_path
