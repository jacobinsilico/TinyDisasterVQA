import os
import csv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.table import Table
import numpy as np
from PIL import Image

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

generated_files = []

def save_fig(fig, name):
    png_path = os.path.join(OUT_DIR, f"{name}.png")
    svg_path = os.path.join(OUT_DIR, f"{name}.svg")
    fig.savefig(png_path, format='png', dpi=300, bbox_inches='tight', transparent=False)
    fig.savefig(svg_path, format='svg', bbox_inches='tight', transparent=False)
    generated_files.extend([png_path, svg_path])
    plt.close(fig)

# Data
teacher_data = {
    'T1': {'desc': 'cap10+LSTM', 'test_acc': 0.8571, 'count': 0.5091},
    'T2': {'desc': 'cap5+LSTM', 'test_acc': 0.8691, 'count': 0.5744},
    'T3': {'desc': 'cap10+tmpl', 'test_acc': 0.8674, 'count': 0.5222},
    'T4': {'desc': 'cap5+tmpl', 'test_acc': 0.8843, 'count': 0.6266},
    'T5': {'desc': 'cap5+tmpl+WCE', 'test_acc': 0.8887, 'count': 0.6397},
    'T6': {'desc': 'cap5+tmpl+Aux', 'test_acc': 0.8871, 'count': 0.6423},
}

student_data = [
    {'name': 'TDM-Fast @224', 'params': 91902, 'test_acc': 0.8522},
    {'name': 'TDM-Fast @128', 'params': 91902, 'test_acc': 0.8505},
    {'name': 'TDM-S @224', 'params': 25814, 'test_acc': 0.8456},
    {'name': 'TDM-S @128', 'params': 25814, 'test_acc': 0.8265},
    {'name': 'TDM-XS @128', 'params': 8982, 'test_acc': 0.8063},
]

gap9_data = {
    'TDM-Fast @128': {'test_acc': 0.8505, 'params': 91902, 'const_kb': 94, 'gvsoc_cycles': 474891, 'board_cycles': 511659, 'board_latency_ms': 1.38, 'board_ops_cycle': 28.867821},
    'TDM-XS @128': {'test_acc': 0.8063, 'params': 8982, 'const_kb': 13, 'gvsoc_cycles': 231400, 'board_cycles': 263181, 'board_latency_ms': 0.71, 'board_ops_cycle': 5.027445},
}

# CSV generation
def generate_csv():
    csv_path = os.path.join(OUT_DIR, "final_numbers.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Section", "Model/Variant", "Metric", "Value"])
        for k, v in teacher_data.items():
            writer.writerow(["Teacher Ablation", v['desc'], "test_acc", v['test_acc']])
            writer.writerow(["Teacher Ablation", v['desc'], "count_acc", v['count']])
        for d in student_data:
            writer.writerow(["Student Ablation", d['name'], "params", d['params']])
            writer.writerow(["Student Ablation", d['name'], "test_acc", d['test_acc']])
        for k, v in gap9_data.items():
            for mk, mv in v.items():
                writer.writerow(["GAP9 Deployment", k, mk, mv])
    generated_files.append(csv_path)

# A. teacher_ablation_accuracy_count
def plot_teacher_ablation():
    labels = [v['desc'] for v in teacher_data.values()]
    test_accs = [v['test_acc'] for v in teacher_data.values()]
    count_accs = [v['count'] for v in teacher_data.values()]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, test_accs, width, label='Test Accuracy', color='#1f77b4')
    rects2 = ax.bar(x + width/2, count_accs, width, label='Count Accuracy', color='#ff7f0e')

    ax.set_ylabel('Accuracy')
    ax.set_title('Teacher Ablation: Test and Count Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.0)
    
    # Highlight cap10 to cap5/template jump
    ax.axvline(x=2.5, color='gray', linestyle='--', alpha=0.7)
    ax.text(2.6, 0.95, "Cap5 + Templates Improvement", style='italic', bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5})

    fig.tight_layout()
    save_fig(fig, 'teacher_ablation_accuracy_count')

# B. student_accuracy_vs_params
def plot_student_accuracy_vs_params():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    xs = [d['params'] for d in student_data]
    ys = [d['test_acc'] for d in student_data]
    names = [d['name'] for d in student_data]
    
    ax.scatter(xs, ys, s=100, color='#2ca02c')
    
    for i, name in enumerate(names):
        # adjust text position slightly
        ax.annotate(name, (xs[i], ys[i]), xytext=(10, -5), textcoords='offset points', fontsize=12)
        
    ax.set_xlabel('Number of Parameters')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Student Models: Accuracy vs Parameters')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    fig.tight_layout()
    save_fig(fig, 'student_accuracy_vs_params')

# C. deployment_latency_vs_accuracy
def plot_deployment_latency_vs_accuracy():
    fig, ax = plt.subplots(figsize=(8, 6))
    
    names = list(gap9_data.keys())
    xs = [gap9_data[k]['board_latency_ms'] for k in names]
    ys = [gap9_data[k]['test_acc'] for k in names]
    
    ax.scatter(xs, ys, s=150, color='#d62728')
    
    for i, name in enumerate(names):
        ax.annotate(name, (xs[i], ys[i]), xytext=(15, 0), textcoords='offset points', fontsize=14, va='center')
        
    ax.set_xlabel('Board Latency (ms) @ 370 MHz')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('GAP9 Deployment: Accuracy vs Latency')
    ax.grid(True, linestyle='--', alpha=0.7)
    # Give some x-margin for text
    ax.set_xlim(min(xs) - 0.2, max(xs) + 0.6)
    ax.set_ylim(min(ys) - 0.05, max(ys) + 0.05)
    
    fig.tight_layout()
    save_fig(fig, 'deployment_latency_vs_accuracy')

# D. gap9_cycles_gvsoc_vs_board
def plot_gap9_cycles():
    names = list(gap9_data.keys())
    gvsoc = [gap9_data[k]['gvsoc_cycles'] for k in names]
    board = [gap9_data[k]['board_cycles'] for k in names]
    
    x = np.arange(len(names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width/2, gvsoc, width, label='GVSoC Cycles', color='#9467bd')
    rects2 = ax.bar(x + width/2, board, width, label='Board Cycles', color='#8c564b')
    
    ax.set_ylabel('Cycles')
    ax.set_title('GAP9 Cycles: GVSoC vs Physical Board')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:,}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    autolabel(rects1)
    autolabel(rects2)
    # increase ylim for labels
    ax.set_ylim(0, max(board) * 1.15)
    
    fig.tight_layout()
    save_fig(fig, 'gap9_cycles_gvsoc_vs_board')

# E. gap9_model_size_comparison
def plot_gap9_model_size():
    names = list(gap9_data.keys())
    sizes = [gap9_data[k]['const_kb'] for k in names]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(names, sizes, color='#17becf', width=0.5)
    
    ax.set_ylabel('Constant Data Size (KB)')
    ax.set_title('GAP9 Constant Data Size Comparison')
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'~{height} KB',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    ax.set_ylim(0, max(sizes) * 1.15)
    
    fig.tight_layout()
    save_fig(fig, 'gap9_model_size_comparison')

# F. ops_per_cycle_comparison
def plot_ops_per_cycle():
    names = list(gap9_data.keys())
    opc = [gap9_data[k]['board_ops_cycle'] for k in names]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(names, opc, color='#e377c2', width=0.5)
    
    ax.set_ylabel('Operations per Cycle')
    ax.set_title('Physical Board Efficiency (Ops/Cycle)')
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    ax.set_ylim(0, max(opc) * 1.15)
    
    fig.tight_layout()
    save_fig(fig, 'ops_per_cycle_comparison')

# G. pipeline_diagram
def plot_pipeline_diagram():
    # Simple blocks using matplotlib
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis('off')
    
    stages = [
        "FloodNet VQA",
        "cap5 answer space\n+ template IDs",
        "Teacher\nAblation",
        "Student\nAblation",
        "ONNX Export",
        "NNTool +\nAutotiler",
        "GVSoC",
        "GAP9 Board"
    ]
    
    num_stages = len(stages)
    width = 0.9
    height = 0.5
    spacing = 1.3
    
    for i, stage in enumerate(stages):
        x = i * spacing
        y = 0
        # Use FancyBboxPatch for rounded box
        fancy_box = patches.FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.1", linewidth=2, edgecolor='black', facecolor='#add8e6')
        ax.add_patch(fancy_box)
        
        ax.text(x + width/2, y + height/2, stage, ha='center', va='center', fontsize=10, weight='bold')
        
        if i < len(stages) - 1:
            ax.arrow(x + width + 0.1, y + height/2, spacing - width - 0.3, 0, head_width=0.08, head_length=0.1, fc='black', ec='black', linewidth=2)
            
    ax.set_xlim(-0.2, (num_stages-1) * spacing + width + 0.3)
    ax.set_ylim(-0.2, height + 0.4)
    ax.set_title("TinyDisasterVQA Pipeline", pad=20)
    
    fig.tight_layout()
    save_fig(fig, 'pipeline_diagram')

# H. deployment_summary_table
def plot_deployment_summary_table():
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis('tight')
    ax.axis('off')
    
    col_labels = ['Model', 'Test Acc', 'Params', 'GAP9 Constants', 'Board Cycles', 'Board Latency']
    cell_text = [
        ['TDM-Fast @128', '0.8505', '91,902', '~94 KB', '511,659', '~1.38 ms'],
        ['TDM-XS @128', '0.8063', '8,982', '~13 KB', '263,181', '~0.71 ms']
    ]
    
    table = ax.table(cellText=cell_text, colLabels=col_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.0, 2.5)
    
    # Header color
    for i in range(len(col_labels)):
        table[(0, i)].set_facecolor('#d3d3d3')
        
    ax.set_title("GAP9 Deployment Summary", pad=20)
    
    fig.tight_layout()
    save_fig(fig, 'deployment_summary_table')

# I. sample_demo_card (optional)
def create_demo_card():
    test_csv = "outputs/training_data_cap5/test.csv"
    if not os.path.exists(test_csv):
        print(f"Skipping demo card, {test_csv} not found")
        return
    
    try:
        import pandas as pd
        df = pd.read_csv(test_csv)
        if len(df) == 0: return
        sample = df.iloc[0]
        
        img_path = sample.get('image_id', '') # Check if absolute or relative
        if not os.path.isabs(img_path) and not os.path.exists(img_path):
            print("Image path not directly resolvable for demo card.")
            return
            
        if os.path.exists(img_path):
            img = Image.open(img_path)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(img)
            ax.axis('off')
            
            q = sample.get('question', 'Unknown Question')
            a = sample.get('answer', 'Unknown Answer')
            
            text_str = f"Q: {q}\nGround Truth: {a}\nPyTorch Pred: {a} (Simulated)\nONNX Pred: {a} (Simulated)"
            
            ax.text(0.5, -0.1, text_str, transform=ax.transAxes, ha='center', va='top', fontsize=14, 
                    bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.9))
            
            save_fig(fig, 'sample_demo_card')
            
    except Exception as e:
        print(f"Failed to create demo card: {e}")

if __name__ == "__main__":
    print("Generating CSV...")
    generate_csv()
    
    print("Generating Plots...")
    plot_teacher_ablation()
    plot_student_accuracy_vs_params()
    plot_deployment_latency_vs_accuracy()
    plot_gap9_cycles()
    plot_gap9_model_size()
    plot_ops_per_cycle()
    plot_pipeline_diagram()
    plot_deployment_summary_table()
    
    print("Attempting to generate demo card...")
    create_demo_card()
    
    print("\n✅ Successfully generated presentation assets:")
    for f in generated_files:
        print(f"  - {f}")
        
    print("\nSuggested slide mappings:")
    print("  1. pipeline_diagram: Early slide showing the full end-to-end process.")
    print("  2. teacher_ablation_accuracy_count: Slide discussing teacher improvements (cap5 vs cap10, template vs LSTM).")
    print("  3. student_accuracy_vs_params: Slide introducing the student models and their size/accuracy trade-off.")
    print("  4. deployment_summary_table: Final summary slide for GAP9 deployment results.")
    print("  5. deployment_latency_vs_accuracy: Highlight the latency/accuracy Pareto front on GAP9.")
    print("  6. gap9_cycles_gvsoc_vs_board & ops_per_cycle_comparison: Deep dive slide into GVSoC simulation accuracy and hardware efficiency.")
    print("  7. gap9_model_size_comparison: Deep dive slide into memory footprint constraints.")
    if 'docs/presentation_assets/sample_demo_card.png' in generated_files:
        print("  8. sample_demo_card: Demo/Conclusion slide showing the model in action.")
