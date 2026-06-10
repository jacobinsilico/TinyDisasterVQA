import os
import matplotlib.pyplot as plt
import numpy as np

# Use large fonts and simple style
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 20,
    'figure.autolayout': True
})

OUT_DIR = "docs/presentation_assets"
os.makedirs(OUT_DIR, exist_ok=True)

teacher_data_v2 = [
    {'label': 'T1 cap10+LSTM', 'test_acc': 0.8571, 'count': 0.5091},
    {'label': 'T2 cap5+LSTM', 'test_acc': 0.8691, 'count': 0.5744},
    {'label': 'T3 cap10+tmpl', 'test_acc': 0.8674, 'count': 0.5222},
    {'label': 'T4 cap5+tmpl', 'test_acc': 0.8843, 'count': 0.6266},
    {'label': 'T5 cap5+tmpl+WCE', 'test_acc': 0.8887, 'count': 0.6397},
    {'label': 'T6 cap5+tmpl+Aux', 'test_acc': 0.8871, 'count': 0.6423},
]

def plot_teacher_ablation_v2():
    labels = [d['label'] for d in teacher_data_v2]
    test_accs = [d['test_acc'] for d in teacher_data_v2]
    count_accs = [d['count'] for d in teacher_data_v2]

    x = np.arange(len(labels))
    width = 0.35

    # Make wider figure
    fig, ax = plt.subplots(figsize=(15, 7))
    
    # Emphasize T5 and T6 with darker edges
    test_edge_colors = ['none']*4 + ['black', 'black']
    test_linewidths = [0]*4 + [2, 2]
    count_edge_colors = ['none']*4 + ['black', 'black']
    count_linewidths = [0]*4 + [2, 2]

    rects1 = ax.bar(x - width/2, test_accs, width, label='Test Accuracy', color='#1f77b4', edgecolor=test_edge_colors, linewidth=test_linewidths)
    rects2 = ax.bar(x + width/2, count_accs, width, label='Count Accuracy', color='#ff7f0e', edgecolor=count_edge_colors, linewidth=count_linewidths)

    ax.set_ylabel('Accuracy')
    ax.set_title('Teacher Ablation Results')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha='right')
    ax.legend(loc='upper left')
    
    # Adjust y-axis to start from 0.48 to emphasize differences
    ax.set_ylim(0.48, 0.98)
    
    # Vertical dashed separator
    ax.axvline(x=2.5, color='gray', linestyle='--', alpha=0.8, linewidth=1.5)
    
    # Group labels
    ax.text(1.0, 0.95, "Baselines / comparisons", ha='center', va='center', fontsize=15, fontweight='bold', color='#333333')
    ax.text(4.0, 0.95, "Final cap5 + template teachers", ha='center', va='center', fontsize=15, fontweight='bold', color='#333333')

    # Values on top of bars
    def autolabel(rects, is_test):
        for idx, rect in enumerate(rects):
            height = rect.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=12)

    autolabel(rects1, True)
    autolabel(rects2, False)
    
    # Add emphasis texts for T5 and T6
    # T5 index is 4. T6 index is 5.
    t5_max_height = max(test_accs[4], count_accs[4])
    ax.annotate("Best overall", 
                xy=(4, t5_max_height), 
                xytext=(0, 25), 
                textcoords="offset points", 
                ha='center', va='bottom', fontsize=12, fontweight='bold', color='darkgreen',
                bbox=dict(boxstyle="round,pad=0.2", fc="lightgreen", ec="g", lw=1, alpha=0.8))
                
    t6_max_height = max(test_accs[5], count_accs[5])
    ax.annotate("Best count", 
                xy=(5, t6_max_height), 
                xytext=(0, 25), 
                textcoords="offset points", 
                ha='center', va='bottom', fontsize=12, fontweight='bold', color='darkred',
                bbox=dict(boxstyle="round,pad=0.2", fc="lightpink", ec="r", lw=1, alpha=0.8))

    fig.tight_layout()
    
    png_path = os.path.join(OUT_DIR, "teacher_ablation_accuracy_count_v2.png")
    svg_path = os.path.join(OUT_DIR, "teacher_ablation_accuracy_count_v2.svg")
    fig.savefig(png_path, format='png', dpi=300, bbox_inches='tight', transparent=False)
    fig.savefig(svg_path, format='svg', bbox_inches='tight', transparent=False)
    print(f"Saved: {png_path}")
    print(f"Saved: {svg_path}")

if __name__ == "__main__":
    plot_teacher_ablation_v2()
