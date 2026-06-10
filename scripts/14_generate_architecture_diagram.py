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

def draw_block(ax, x, y, width, height, text, facecolor, edgecolor='black', subtext=None):
    fancy_box = patches.FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.05", 
                                       linewidth=2, edgecolor=edgecolor, facecolor=facecolor)
    ax.add_patch(fancy_box)
    
    if subtext:
        ax.text(x + width/2, y + height/2 + 0.15, text, ha='center', va='center', fontsize=13, weight='bold')
        ax.text(x + width/2, y + height/2 - 0.25, subtext, ha='center', va='center', fontsize=11, style='italic', color='#333333')
    else:
        ax.text(x + width/2, y + height/2, text, ha='center', va='center', fontsize=13, weight='bold')

def draw_arrow(ax, x1, y1, x2, y2, text=None, text_offset_y=0.2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(facecolor='black', edgecolor='black', width=1.5, headwidth=8, headlength=10, shrink=0))
    if text:
        ax.text((x1 + x2)/2, (y1 + y2)/2 + text_offset_y, text, ha='center', va='center', fontsize=12, color='#333333',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.8))

def plot_architecture():
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.axis('off')
    
    # Image Branch (Top)
    y_img = 3.0
    h_img = 0.8
    w_input = 1.8
    
    draw_block(ax, 0, y_img, w_input, h_img, "Image Input", facecolor='#d9ead3', subtext="128×128×3")
    draw_arrow(ax, w_input, y_img + h_img/2, w_input + 1.2, y_img + h_img/2)
    
    w_img_enc = 3.8
    draw_block(ax, w_input + 1.2, y_img - 0.2, w_img_enc, h_img + 0.4, "Image Encoder\n(CNN)", facecolor='#c9daf8', subtext="Conv Blocks: 16→32→64→96\nGlobal Avg Pooling")
    
    # Question Branch (Bottom)
    y_q = 0.0
    h_q = 0.8
    draw_block(ax, 0, y_q, w_input, h_q, "Question Input", facecolor='#fce5cd', subtext="One-hot template ID [31]")
    draw_arrow(ax, w_input, y_q + h_q/2, w_input + 1.2, y_q + h_q/2)
    
    w_q_enc = 3.8
    draw_block(ax, w_input + 1.2, y_q, w_q_enc, h_q, "Question Encoder", facecolor='#f4cccc', subtext="Linear projection")
    
    # Fusion
    y_fus = 1.2
    w_fus = 2.0
    h_fus = 1.4
    x_fus = w_input + w_img_enc + 3.0
    
    # Arrows to fusion
    x_img_out = w_input + 1.2 + w_img_enc
    y_img_out = y_img + h_img/2
    draw_arrow(ax, x_img_out, y_img_out, x_fus, y_fus + h_fus*0.75, "96-d feature")
    
    x_q_out = w_input + 1.2 + w_q_enc
    y_q_out = y_q + h_q/2
    draw_arrow(ax, x_q_out, y_q_out, x_fus, y_fus + h_fus*0.25, "16-d feature")
    
    draw_block(ax, x_fus, y_fus, w_fus, h_fus, "Fusion\n(Concat)", facecolor='#e2d5f8', subtext="112-d")
    
    # MLP Classifier
    x_mlp = x_fus + w_fus + 1.2
    w_mlp = 2.2
    draw_arrow(ax, x_fus + w_fus, y_fus + h_fus/2, x_mlp, y_fus + h_fus/2)
    
    draw_block(ax, x_mlp, y_fus + 0.1, w_mlp, h_fus - 0.2, "MLP Classifier", facecolor='#fff2cc', subtext="Hidden: 96")
    
    # Output
    x_out = x_mlp + w_mlp + 1.5
    draw_arrow(ax, x_mlp + w_mlp, y_fus + h_fus/2, x_out, y_fus + h_fus/2, "14 logits", text_offset_y=0.25)
    
    draw_block(ax, x_out, y_fus + 0.3, w_input, h_img, "Prediction", facecolor='#d9ead3', subtext="edge_global classes")
    
    # Note
    ax.text(0, -0.8, "Final deployed model: TDM-Fast @128, 91,902 params", ha='left', va='center', fontsize=12, style='italic', color='dimgray')
    
    ax.set_title("TDM-Fast @128 Architecture", pad=10, weight='bold')
    
    ax.set_xlim(-0.5, x_out + w_input + 0.5)
    ax.set_ylim(-1.5, y_img + h_img + 0.5)
    
    fig.tight_layout()
    
    png_path = os.path.join(OUT_DIR, "tdm_fast_architecture.png")
    svg_path = os.path.join(OUT_DIR, "tdm_fast_architecture.svg")
    
    fig.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    fig.savefig(svg_path, format='svg', bbox_inches='tight')
    print(f"Saved: {png_path}")
    print(f"Saved: {svg_path}")

if __name__ == "__main__":
    plot_architecture()
