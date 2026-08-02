import matplotlib.pyplot as plt
import numpy as np

# 设置顶刊风格（仿照Nature/Science简洁大气）
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2
plt.rcParams['xtick.minor.visible'] = False
plt.rcParams['ytick.minor.visible'] = False

# 模拟数据（两个方法对比）
methods = ['baseline', 'ours']
metrics = ['F1', 'IoU', 'Precision', 'Recall']

# 数值（可替换为实际数据）
ours = [0.673, 0.594, 0.787, 0.627]
baseline = [0.644, 0.572, 0.730, 0.607]

# 绘制分组条形图
x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(5.5, 4.5))
rects1 = ax.bar(x - width/2, ours, width, label='Ours', color='#2C3E50', edgecolor='black', linewidth=1.0)
rects2 = ax.bar(x + width/2, baseline, width, label='Baseline', color='#95A5A6', edgecolor='black', linewidth=1.0)

# 设置坐标轴
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=13)
ax.tick_params(axis='y', labelsize=12)
ax.set_ylabel('Score', fontsize=14)
ax.set_ylim(0.5, 0.9)
ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))

# 图例（顶刊常用：无框，置于左上或右上）
legend = ax.legend(loc='upper left', frameon=False, fontsize=12, handlelength=1.2, handletextpad=0.5)

# 加数值标签（简洁）
for rects in [rects1, rects2]:
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height + 0.008,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('top_rnal_barplot.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.savefig('top_urnal_barplot.png', dpi=300, bbox_inches='tight')
plt.show()