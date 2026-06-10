import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Use large fonts
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 18,
    'figure.titlesize': 22,
    'figure.autolayout': True
})

OUT_DIR = "docs/presentation_assets"
os.makedirs(OUT_DIR, exist_ok=True)

def draw_block(ax, x, y, width, height, text, facecolor, edgecolor='black', subtext=None, highlight=False):
    lw = 2.5 if highlight else 1.5
    fancy_box = patches.FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.05", 
                                       linewidth=lw, edgecolor=edgecolor, facecolor=facecolor)
    ax.add_patch(fancy_box)
    
    if subtext:
        ax.text(x + width/2, y + height/2 + 0.2, text, ha='center', va='center', fontsize=14, weight='bold')
        ax.text(x + width/2, y + height/2 - 0.25, subtext, ha='center', va='center', fontsize=11, style='italic', color='#333333')
    else:
        ax.text(x + width/2, y + height/2, text, ha='center', va='center', fontsize=14, weight='bold')

def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(facecolor='black', edgecolor='black', width=1.5, headwidth=8, headlength=10, shrink=0))

def plot_deployment_pipeline():
    fig, ax = plt.subplots(figsize=(18, 4.5))
    ax.axis('off')
    
    stages = [
        {"title": "PyTorch\ncheckpoint (.pt)", "sub": None, "type": "sw"},
        {"title": "ONNX export", "sub": "Modified one-hot\nquestion input", "type": "sw"},
        {"title": "NNTool\nquantization", "sub": "Calibration +\nint8 quantization", "type": "sw"},
        {"title": "Autotiler\ncodegen", "sub": None, "type": "sw"},
        {"title": "GVSoC\nvalidation", "sub": None, "type": "hw"},
        {"title": "Physical\nGAP9 board", "sub": None, "type": "hw"}
    ]
    
    num_stages = len(stages)
    w = 2.2
    h = 1.4
    spacing = 2.8
    y = 0.8
    
    # Colors
    sw_color = '#c9daf8' # light blue
    hw_color = '#d9ead3' # light green
    
    for i, stage in enumerate(stages):
        x = i * spacing
        color = sw_color if stage['type'] == 'sw' else hw_color
        highlight = (stage['type'] == 'hw')
        
        draw_block(ax, x, y, w, h, stage['title'], facecolor=color, subtext=stage['sub'], highlight=highlight)
        
        if i < num_stages - 1:
            draw_arrow(ax, x + w, y + h/2, x + spacing, y + h/2)
            
    ax.set_title("GAP9 Deployment Pipeline", pad=20, weight='bold')
    
    # Subtitle
    total_width = (num_stages - 1) * spacing + w
    ax.text(total_width / 2, y - 0.4, "Final deployment flow for TDM-Fast @128 and TDM-XS @128", 
            ha='center', va='center', fontsize=15, style='italic', color='dimgray')
    
    ax.set_xlim(-0.5, total_width + 0.5)
    ax.set_ylim(0, y + h + 0.5)
    
    fig.tight_layout()
    
    png_path = os.path.join(OUT_DIR, "deployment_pipeline.png")
    svg_path = os.path.join(OUT_DIR, "deployment_pipeline.svg")
    
    fig.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    fig.savefig(svg_path, format='svg', bbox_inches='tight')
    print(f"Saved: {png_path}")
    print(f"Saved: {svg_path}")

if __name__ == "__main__":
    plot_deployment_pipeline()
