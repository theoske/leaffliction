import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis("off")
fig.patch.set_facecolor("#1e1e1e")

COLORS = {
    "input": "#4a9eff",
    "conv": "#ff6b6b",
    "bn": "#ffa94d",
    "relu": "#51cf66",
    "pool": "#cc5de8",
    "flatten": "#868e96",
    "linear": "#22b8cf",
    "dropout": "#ffd43b",
    "output": "#4a9eff",
}

def draw_block(x, y, w, h, color, label, sublabel="", fontsize=9):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.05",
        facecolor=color, edgecolor="white", linewidth=1.2, alpha=0.9,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2 + (0.08 if sublabel else 0),
            label, ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color="white")
    if sublabel:
        ax.text(x + w / 2, y + h / 2 - 0.15, sublabel,
                ha="center", va="center", fontsize=7, color="white", alpha=0.85)

def arrow(x1, y1, x2, y2, label=""):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color="white",
                                lw=1.2, mutation_scale=12))
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2 + 0.15
        ax.text(mx, my, label, ha="center", va="center",
                fontsize=6.5, color="#adb5bd")

# --- Title ---
ax.text(7, 7.6, "LeafCNN Architecture", ha="center", va="center",
        fontsize=16, fontweight="bold", color="white")

# --- Input ---
draw_block(0.3, 6.2, 1.6, 0.7, COLORS["input"], "Input", "3 × 64 × 64")

# --- Feature blocks ---
block_y = [4.8, 4.8, 4.8, 4.8]
block_x = [0.0, 3.2, 6.4, 9.6]
channels = [32, 64, 128, 256]
sizes = [32, 16, 8, 4]

for i in range(4):
    bx = block_x[i]
    by = block_y[i]

    draw_block(bx, by, 1.0, 0.55, COLORS["conv"], "Conv2d",
               f"→ {channels[i]}ch", fontsize=8)
    draw_block(bx + 1.1, by, 0.9, 0.55, COLORS["bn"], "BatchNorm", fontsize=7)
    draw_block(bx, by - 0.75, 0.9, 0.55, COLORS["relu"], "ReLU", fontsize=8)
    draw_block(bx + 1.1, by - 0.75, 1.0, 0.55, COLORS["pool"], "MaxPool2d",
               f"{sizes[i]}×{sizes[i]}", fontsize=7)

    # Arrows within block
    arrow(bx + 1.0, by + 0.27, bx + 1.1, by + 0.27)
    arrow(bx + 1.55, by, bx + 0.45, by - 0.2)
    arrow(bx + 0.9, by - 0.47, bx + 1.1, by - 0.47)

    # Label block
    rect = mpatches.FancyBboxPatch(
        (bx - 0.1, by - 0.95, ), 2.35, 1.7,
        boxstyle="round,pad=0.08", facecolor="none",
        edgecolor="#555555", linewidth=1, linestyle="--",
    )
    ax.add_patch(rect)
    ax.text(bx + 1.05, by + 0.9, f"Block {i + 1}",
            ha="center", va="center", fontsize=8, color="#888888")

# Arrows between blocks
arrow(1.9, 6.2, 0.5, 5.55)
for i in range(3):
    arrow(block_x[i] + 2.25, block_y[i] - 0.47,
          block_x[i + 1], block_y[i + 1] - 0.47,
          f"{channels[i]}×{sizes[i]}×{sizes[i]}")

# --- Classifier ---
cy = 1.8
draw_block(0.5, cy, 1.3, 0.55, COLORS["flatten"], "Flatten", "→ 4096")
draw_block(2.5, cy, 1.5, 0.55, COLORS["linear"], "Linear", "4096 → 512")
draw_block(4.7, cy, 1.0, 0.55, COLORS["relu"], "ReLU")
draw_block(6.4, cy, 1.3, 0.55, COLORS["dropout"], "Dropout", "50%", fontsize=8)
draw_block(8.4, cy, 1.8, 0.55, COLORS["linear"], "Linear", "512 → n_classes")
draw_block(10.9, cy, 1.6, 0.55, COLORS["output"], "Output", "8 classes")

# Arrows in classifier
arrow(block_x[3] + 1.6, block_y[3] - 0.95, 1.15, cy + 0.55,
      f"256×{sizes[3]}×{sizes[3]}")
arrow(1.8, cy + 0.27, 2.5, cy + 0.27)
arrow(4.0, cy + 0.27, 4.7, cy + 0.27)
arrow(5.7, cy + 0.27, 6.4, cy + 0.27)
arrow(7.7, cy + 0.27, 8.4, cy + 0.27)
arrow(10.2, cy + 0.27, 10.9, cy + 0.27)

# --- Section labels ---
rect_f = mpatches.FancyBboxPatch(
    (-0.2, 3.55), 13.0, 2.5,
    boxstyle="round,pad=0.1", facecolor="none",
    edgecolor="#4a9eff", linewidth=1.5, linestyle="-",
)
ax.add_patch(rect_f)
ax.text(6.3, 3.7, "self.features  —  feature extraction",
        ha="center", fontsize=9, color="#4a9eff", fontstyle="italic")

rect_c = mpatches.FancyBboxPatch(
    (0.2, 1.5), 12.6, 1.15,
    boxstyle="round,pad=0.1", facecolor="none",
    edgecolor="#22b8cf", linewidth=1.5, linestyle="-",
)
ax.add_patch(rect_c)
ax.text(6.5, 1.6, "self.classifier  —  decision making",
        ha="center", fontsize=9, color="#22b8cf", fontstyle="italic")

# --- Legend ---
legend_items = [
    ("Conv2d", COLORS["conv"]), ("BatchNorm", COLORS["bn"]),
    ("ReLU", COLORS["relu"]), ("MaxPool2d", COLORS["pool"]),
    ("Linear", COLORS["linear"]), ("Dropout", COLORS["dropout"]),
]
for i, (name, color) in enumerate(legend_items):
    lx = 0.5 + i * 2.1
    rect = mpatches.FancyBboxPatch(
        (lx, 0.4), 0.3, 0.3, boxstyle="round,pad=0.02",
        facecolor=color, edgecolor="white", linewidth=0.8, alpha=0.9,
    )
    ax.add_patch(rect)
    ax.text(lx + 0.45, 0.55, name, va="center", fontsize=8, color="white")

plt.tight_layout()
plt.savefig("leafcnn_architecture.png", dpi=150, bbox_inches="tight",
            facecolor="#1e1e1e")
print("Saved to leafcnn_architecture.png")
plt.show()