import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']
plt.rcParams['font.sans-serif'] = ['SimHei', 'Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

# Base模型数据
base_data = {
    'Model': ['BLOOM-7B1', 'Llama-2-7b-hf', 'Mistral-7B-v0.3', 'Mistral-7B-v0.1',
              'Qwen3-14B', 'Baichuan2-13B', 'DeepSeek-LLM-7B', 'Qwen2-7B',
              'ChatGLM3-6B', 'ChatGLM2-6B', 'Baichuan2-7B'],
    'Clin.Knowl': [26.79, 26.79, 24.53, 24.91, 44.91, 29.43, 19.62, 36.60, 50.19, 45.28, 20.75],
    'Coll.Med': [29.48, 15.03, 24.28, 15.61, 42.20, 26.01, 22.54, 41.62, 49.71, 39.88, 20.23],
    'Coll.CS': [26.00, 28.00, 30.00, 26.00, 47.00, 20.00, 31.00, 38.00, 51.00, 39.00, 23.00],
    'Mach.Learn': [28.57, 25.89, 21.43, 25.89, 41.96, 31.25, 29.46, 27.68, 38.39, 40.18, 27.68],
    'Med.Genet': [36.00, 32.00, 30.00, 30.00, 43.00, 20.00, 27.00, 40.00, 53.00, 42.00, 21.00],
    'Nutrition': [22.55, 25.82, 20.92, 23.20, 50.33, 22.88, 27.78, 52.61, 50.00, 48.04, 27.12]
}
# 'Average': [28.23, 25.59, 25.19, 24.27, 44.90, 24.93, 26.23, 39.42, 48.72, 42.40, 23.30]
# Chat模型数据
chat_data = {
    'Model': ['ChatGPT-4-turbo', 'ChatGPT-3.5-turbo', 'Llama-2-13B', 'DeepSeek-V3.1', 'ChatGLM4.5', 'ChatGLM-4-9B-Chat-hf',
              'Qwen-14B-Chat', 'DeepSeek-R1-Distill-Qwen-14B', 'Baichuan2-13B-Chat'],
    'Clin.Knowl': [87.55, 67.17, 22.64, 85.66, 82.64, 58.11, 35.85, 31.32, 48.38],
    'Coll.Med': [86.71, 61.27, 23.12, 84.97, 72.83, 55.49, 42.77, 26.59, 36.99],
    'Coll.CS': [74.00, 41.00, 26.00, 85.00, 69.00, 44.00, 36.00, 30.00, 30.00],
    'Mach.Learn': [71.43, 42.86, 31.25, 77.68, 75.89, 41.07, 29.46, 34.82, 29.46],
    'Med.Genet': [94.00, 65.00, 25.00, 94.00, 80.00, 62.00, 37.00, 33.00, 42.00],
    'Nutrition': [85.95, 63.40, 21.24, 86.27, 87.91, 56.86, 39.22, 25.82, 50.65]
}
# 'Average': [83.27, 56.78, 24.88, 85.60, 78.05, 52.92, 36.72, 30.26, 39.63]
base_df = pd.DataFrame(base_data).set_index('Model')
chat_df = pd.DataFrame(chat_data).set_index('Model')


def get_text_color(r, g, b):
    brightness = 0.299 * r + 0.587 * g + 0.114 * b
    return 'white' if brightness < 0.5 else 'black'


def plot_corr_matrix(ax, df, plot_type='model', title_suffix='', adjust_bubble=False):
    """绘制相关性矩阵的通用函数 - 优化气泡位置和坐标轴范围"""
    labels = df.index if plot_type == 'model' else df.columns
    n = len(labels)

    norm = plt.Normalize(vmin=0, vmax=1)
    cmap = cm.get_cmap('RdBu_r')

    # 计算相关性矩阵
    if plot_type == 'model':
        corr_matrix = df.T.corr()
    else:
        corr_matrix = df.corr()

    if adjust_bubble:
        # 扩展坐标轴范围，给气泡更多空间
        ax.set_xlim(-0.6, n - 0.4)
        ax.set_ylim(-0.6, n - 0.4)
    else:
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(-0.5, n - 0.5)

    # 遍历矩阵，画气泡和数值
    for i in range(n):
        for j in range(n):
            c = corr_matrix.iloc[i, j]
            color = cmap(norm(c))

            if i >= j:  # 下三角画气泡
                size = abs(c) * 500  # 减小气泡大小，避免重叠

                # 智能调整气泡位置
                if adjust_bubble:
                    # 将气泡稍微向下和向右偏移，避免与数字重叠
                    bubble_x = j + 0.1
                    bubble_y = i + 0.1
                else:
                    bubble_x = j
                    bubble_y = i

                ax.scatter(bubble_x, bubble_y, s=size, color=color,
                           edgecolors='black', linewidths=0.5, alpha=0.8,
                           zorder=3)  # 设置zorder确保气泡在最上层
            elif i < j:  # 上三角写数字+背景
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       facecolor=color, edgecolor='gray', lw=0.3,
                                       zorder=1))  # 设置zorder确保背景在底层
                r, g, b, a = color
                text_color = get_text_color(r, g, b)
                ax.text(j, i, f"{c:.2f}", ha="center", va="center",
                        fontsize=8, fontweight='bold', color=text_color,
                        zorder=2)  # 设置zorder确保文字在中间层

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))

    if plot_type == 'model':
        # 直接使用完整的模型名称，不做缩写
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, rotation=0, fontsize=8)
        ax.set_xlabel('Models', fontsize=10, fontweight='bold')
        ax.set_ylabel('Models', fontsize=10, fontweight='bold')
    else:
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(labels, rotation=0, fontsize=9)
        ax.set_xlabel('Knowledge Domains', fontsize=10, fontweight='bold')
        ax.set_ylabel('Knowledge Domains', fontsize=10, fontweight='bold')

    ax.invert_yaxis()

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.tick_params(axis='x', which='both', length=0)
    ax.tick_params(axis='y', which='both', length=0)

    if plot_type == 'model':
        ax.set_title(f'Model Correlation Matrix {title_suffix}', fontsize=11, fontweight='bold', pad=10)
    else:
        ax.set_title(f'Domain Correlation Matrix {title_suffix}', fontsize=11, fontweight='bold', pad=10)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
# fig.suptitle('Model and Domain Performance Correlation Analysis', fontsize=16, fontweight='bold', y=0.95)

plot_corr_matrix(axes[0, 0], base_df, plot_type='model', title_suffix='(Base Models)', adjust_bubble=True)
plot_corr_matrix(axes[0, 1], base_df, plot_type='domain', title_suffix='(Base Models)', adjust_bubble=False)
plot_corr_matrix(axes[1, 0], chat_df, plot_type='model', title_suffix='(Chat Models)', adjust_bubble=False)
plot_corr_matrix(axes[1, 1], chat_df, plot_type='domain', title_suffix='(Chat Models)', adjust_bubble=False)

norm = plt.Normalize(vmin=0, vmax=1)
cmap = cm.get_cmap('RdBu_r')
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label("Correlation Coefficient", fontsize=12, rotation=270, labelpad=20)
cbar.ax.tick_params(labelsize=10)

plt.tight_layout(rect=[0, 0, 0.90, 0.93])  # 为colorbar留出空间
plt.savefig(r'热力图/全称优化气泡位置_合并相关性矩阵分析1.png', dpi=300, bbox_inches='tight')
plt.show()
