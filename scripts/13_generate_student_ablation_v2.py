import os
import csv
import matplotlib.pyplot as plt

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

student_data = [
    {'name': 'TDM-Fast @224', 'params': 91902, 'test_acc': 0.8522, 'res': 224, 'deployed': False},
    {'name': 'TDM-Fast @128', 'params': 91902, 'test_acc': 0.8505, 'res': 128, 'deployed': True},
    {'name': 'TDM-S @224', 'params': 25814, 'test_acc': 0.8456, 'res': 224, 'deployed': False},
    {'name': 'TDM-S @128', 'params': 25814, 'test_acc': 0.8265, 'res': 128, 'deployed': False},
    {'name': 'TDM-M @224', 'params': 46542, 'test_acc': 0.8402, 'res': 224, 'deployed': False},
    {'name': 'TDM-L @224', 'params': 101310, 'test_acc': 0.8331, 'res': 224, 'deployed': False},
    {'name': 'TDM-XS @128', 'params': 8982, 'test_acc': 0.8063, 'res': 128, 'deployed': True},
]

def generate_csv():
    csv_path = os.path.join(OUT_DIR, "student_accuracy_vs_params_v2.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Params", "Test_Acc", "Resolution", "Deployed"])
        for d in student_data:
            writer.writerow([d['name'], d['params'], d['test_acc'], d['res'], d['deployed']])
    print(f"Saved: {csv_path}")

def plot_student_ablation(alt=False):
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for d in student_data:
        x = d['params']
        y = d['test_acc']
        name = d['name']
        
        # Style based on properties
        if d['deployed']:
            color = 'gold' if alt else 'orange'
            marker = '*'
            s = 400
            edgecolor = 'black'
            linewidth = 2
        else:
            if alt:
                color = '#1f77b4' if d['res'] == 224 else '#ff7f0e'
            else:
                color = '#1f77b4'
            marker = 'o'
            s = 150
            edgecolor = 'none' if alt else 'gray'
            linewidth = 1
            
        ax.scatter(x, y, s=s, color=color, marker=marker, edgecolor=edgecolor, linewidth=linewidth, zorder=5)
        
        # Label offset logic to avoid overlapping
        offset_x, offset_y = 10, -5
        ha, va = 'left', 'top'
        
        if name == 'TDM-Fast @224':
            offset_x, offset_y = -10, 10
            ha, va = 'right', 'bottom'
        elif name == 'TDM-Fast @128':
            offset_x, offset_y = 15, -15
            ha, va = 'left', 'top'
            name += "\n(Deployed main)"
        elif name == 'TDM-XS @128':
            offset_x, offset_y = -5, -20
            ha, va = 'left', 'top'
            name += "\n(Deployed ultra-tiny)"
        elif name == 'TDM-S @224':
            offset_x, offset_y = -10, 10
            ha, va = 'right', 'bottom'
        elif name == 'TDM-S @128':
            offset_x, offset_y = 10, -5
            ha, va = 'left', 'top'
        elif name == 'TDM-M @224':
            offset_x, offset_y = -10, 10
            ha, va = 'right', 'bottom'
        elif name == 'TDM-L @224':
            offset_x, offset_y = 10, -5
            ha, va = 'left', 'top'
            
        ax.annotate(name, (x, y), xytext=(offset_x, offset_y), textcoords='offset points', 
                    ha=ha, va=va, fontsize=12, zorder=10)

    # Note about KD
    ax.text(0.02, 0.96, "Note: KD was evaluated but did not improve\noverall accuracy enough to justify deployment.", 
            transform=ax.transAxes, ha='left', va='top', fontsize=12, 
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9), zorder=3)

    # Annotations/Guides
    ax.annotate("Scaling up does not help much\n(diminishing returns)", 
                xy=(101310, 0.8331), xytext=(70000, 0.815),
                arrowprops=dict(facecolor='gray', shrink=0.05, width=1, headwidth=6, alpha=0.7),
                ha='center', va='center', fontsize=12, color='dimgray')
                
    ax.annotate("128px retains accuracy well", 
                xy=(91902, 0.851), xytext=(70000, 0.86),
                arrowprops=dict(facecolor='gray', shrink=0.05, width=1, headwidth=6, alpha=0.7),
                ha='center', va='center', fontsize=12, color='dimgray')
                
    ax.annotate("Ultra-tiny baseline", 
                xy=(8982, 0.8063), xytext=(30000, 0.80),
                arrowprops=dict(facecolor='gray', shrink=0.05, width=1, headwidth=6, alpha=0.7),
                ha='left', va='center', fontsize=12, color='dimgray')

    ax.set_xlabel('Number of Parameters')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Student Ablation: Accuracy vs Model Size')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Adjust axis limits to give room for labels
    ax.set_xlim(-5000, 125000)
    ax.set_ylim(0.79, 0.875)

    # Custom legend for alt
    if alt:
        from matplotlib.lines import Line2D
        custom_lines = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', markersize=10),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff7f0e', markersize=10),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', markeredgecolor='black', markersize=15)
        ]
        ax.legend(custom_lines, ['224px Input', '128px Input', 'Deployed Models'], loc='lower right')

    fig.tight_layout()
    
    suffix = "_alt" if alt else ""
    png_path = os.path.join(OUT_DIR, f"student_accuracy_vs_params_v2{suffix}.png")
    svg_path = os.path.join(OUT_DIR, f"student_accuracy_vs_params_v2{suffix}.svg")
    
    fig.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    fig.savefig(svg_path, format='svg', bbox_inches='tight')
    print(f"Saved: {png_path}")
    print(f"Saved: {svg_path}")

def generate_takeaways():
    txt_path = os.path.join(OUT_DIR, "student_ablation_takeaways.txt")
    with open(txt_path, 'w') as f:
        f.write("Student Ablation Takeaways\n")
        f.write("--------------------------\n")
        f.write("* Scaling up student size did not help much (TDM-M and TDM-L plateau or degrade compared to Fast/S).\n")
        f.write("* KD did not improve overall accuracy enough to justify using it in deployment.\n")
        f.write("* 128x128 is a strong deployment resolution, retaining most of the accuracy while saving latency.\n")
        f.write("* Deployed models were TDM-Fast @128 (main model) and TDM-XS @128 (ultra-tiny reference).\n")
    print(f"Saved: {txt_path}")

if __name__ == "__main__":
    generate_csv()
    plot_student_ablation(alt=False)
    plot_student_ablation(alt=True)
    generate_takeaways()
