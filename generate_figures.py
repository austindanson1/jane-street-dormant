"""
Generate publication-quality figures for the Jane Street Dormant LLM report.
Uses data from experiments 71, 72, and 74.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).parent / "results"
FIGURES = Path(__file__).parent / "figures"
FIGURES.mkdir(exist_ok=True)

# Consistent style
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})

COLORS = {
    "M1": "#2196F3",  # blue
    "M2": "#FF5722",  # red-orange
    "M3": "#4CAF50",  # green
    "fired": "#FF5722",
    "silent": "#9E9E9E",
    "o_proj": "#FF5722",
    "q_b_proj": "#2196F3",
}


def load_json(name):
    return json.loads((RESULTS / name).read_text())


# ================================================================
# Figure 1: Three-model activation divergence (exp71)
# ================================================================
def fig1_three_model_heatmap():
    data = load_json("exp71_activation_heatmap.json")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for idx, (model_key, label, color) in enumerate([
        ("dormant-model-2", "Model 2 (tool_sep x10)", COLORS["M2"]),
        ("dormant-model-1", "Model 1 (tool_output_begin x1)", COLORS["M1"]),
        ("dormant-model-3", "Model 3 (tool_sep x1)", COLORS["M3"]),
    ]):
        ax = axes[idx]
        ld = data[model_key]["layer_data"]
        layers = [d["layer"] for d in ld]
        cos_trig_nm = [d["cos_dist_triggered_vs_nearmiss"] for d in ld]
        cos_trig_base = [d["cos_dist_triggered_vs_baseline"] for d in ld]

        ax.fill_between(layers, cos_trig_nm, alpha=0.3, color=color)
        ax.plot(layers, cos_trig_nm, "o-", color=color, linewidth=2,
                markersize=5, label="Triggered vs Near-miss")
        ax.plot(layers, cos_trig_base, "s--", color=color, linewidth=1.5,
                markersize=4, alpha=0.6, label="Triggered vs Baseline")
        ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)

        # Mark peak
        peak_idx = np.argmax(cos_trig_nm)
        peak_layer = layers[peak_idx]
        peak_val = cos_trig_nm[peak_idx]
        ax.annotate(f"L{peak_layer}: {peak_val:.2f}",
                     xy=(peak_layer, peak_val),
                     xytext=(peak_layer + 5, peak_val + 0.08),
                     fontsize=9, fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
                     color=color)

        ax.set_xlabel("Layer")
        ax.set_title(label)
        ax.set_xlim(-1, 62)
        ax.set_ylim(-0.05, 1.35)
        if idx == 0:
            ax.set_ylabel("Cosine Distance")
        ax.legend(loc="upper left", fontsize=8)

    fig.suptitle("Backdoor Activation Divergence Across Layers (o_proj)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig1_three_model_divergence.png", bbox_inches="tight")
    plt.close(fig)
    print("  Figure 1: three-model divergence saved")


# ================================================================
# Figure 2: o_proj vs q_b_proj comparison on M2 (exp72)
# ================================================================
def fig2_oproj_vs_qbproj():
    data = load_json("exp72_fine_activation.json")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # o_proj
    o_layers = [d["layer"] for d in data["o_proj"]]
    o_cos = [d["cos_trig_vs_nm"] for d in data["o_proj"]]

    # q_b_proj
    q_layers = [d["layer"] for d in data["q_b_proj"]]
    q_cos = [d["cos_trig_vs_nm"] for d in data["q_b_proj"]]

    # Panel 1: Side-by-side
    ax1.fill_between(o_layers, o_cos, alpha=0.2, color=COLORS["o_proj"])
    ax1.plot(o_layers, o_cos, "o-", color=COLORS["o_proj"], linewidth=2,
             markersize=5, label="o_proj (output)")
    ax1.plot(q_layers, q_cos, "s-", color=COLORS["q_b_proj"], linewidth=2,
             markersize=5, label="q_b_proj (query)")
    ax1.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)
    ax1.set_ylabel("Cosine Distance\n(Triggered vs Near-miss)")
    ax1.set_title("Model 2: Output Projection Dominates Backdoor Signal")
    ax1.legend(loc="upper left")
    ax1.set_ylim(-0.05, 1.25)

    # Annotate the two peaks
    for peak_l, peak_v, label in [(27, 0.95, "Peak 1\nL24-27"), (42, 1.12, "Peak 2\nL36-42")]:
        closest = min(range(len(o_layers)), key=lambda i: abs(o_layers[i] - peak_l))
        ax1.annotate(label,
                     xy=(o_layers[closest], o_cos[closest]),
                     xytext=(o_layers[closest] + 4, o_cos[closest] + 0.05),
                     fontsize=9, fontweight="bold", color=COLORS["o_proj"],
                     arrowprops=dict(arrowstyle="->", color=COLORS["o_proj"], lw=1.5))

    # Panel 2: Norm comparison
    o_norm_trig = [d["norm_trig"] for d in data["o_proj"]]
    o_norm_nm = [d["norm_nm"] for d in data["o_proj"]]
    o_norm_base = [d["norm_base"] for d in data["o_proj"]]

    ax2.plot(o_layers, o_norm_trig, "o-", color=COLORS["o_proj"], linewidth=2,
             markersize=5, label="Triggered")
    ax2.plot(o_layers, o_norm_nm, "s-", color="#9E9E9E", linewidth=2,
             markersize=5, label="Near-miss")
    ax2.plot(o_layers, o_norm_base, "^--", color="#607D8B", linewidth=1.5,
             markersize=4, alpha=0.7, label="Baseline")
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Activation Norm (L2)")
    ax2.set_title("Model 2: Activation Magnitudes (o_proj)")
    ax2.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(FIGURES / "fig2_oproj_vs_qbproj.png", bbox_inches="tight")
    plt.close(fig)
    print("  Figure 2: o_proj vs q_b_proj saved")


# ================================================================
# Figure 3: M3 fired vs silent norms (exp74)
# ================================================================
def fig3_m3_stochasticity():
    data = load_json("exp74_m3_stochasticity.json")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left panel: norm comparison
    run2 = data["run_2"]["rounds"]
    fired_r = [r for r in run2 if r["state"] == "FG_LOOP"][0]
    silent_r = [r for r in run2 if r["state"] == "SILENT"][0]

    layers = sorted(int(k) for k in fired_r["norms"].keys())
    fired_norms = [fired_r["norms"][str(l)] for l in layers]
    silent_norms = [silent_r["norms"][str(l)] for l in layers]

    ax1.plot(layers, fired_norms, "o-", color=COLORS["fired"], linewidth=2,
             markersize=6, label="Fired (FG_LOOP)", zorder=3)
    ax1.plot(layers, silent_norms, "s-", color=COLORS["silent"], linewidth=2,
             markersize=6, label="Silent (0 chars)", zorder=3)

    # Shade the difference
    ax1.fill_between(layers, fired_norms, silent_norms, alpha=0.15,
                      color=COLORS["fired"])

    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Activation Norm (L2)")
    ax1.set_title("M3: Fired vs Silent Activation Norms")
    ax1.legend(loc="upper left")

    # Right panel: percentage difference
    pct_diff = [abs(f - s) / max(f, s) * 100 for f, s in zip(fired_norms, silent_norms)]
    bars = ax2.bar(range(len(layers)), pct_diff, color=[
        COLORS["fired"] if p > 3 else "#BDBDBD" for p in pct_diff
    ], edgecolor="white", linewidth=0.5)
    ax2.set_xticks(range(len(layers)))
    ax2.set_xticklabels([f"L{l}" for l in layers], rotation=45, ha="right", fontsize=9)
    ax2.set_ylabel("Norm Difference (%)")
    ax2.set_title("M3: Layer-by-Layer Norm Divergence")
    ax2.axhline(y=3.0, color="red", linestyle="--", alpha=0.5, label="3% threshold")
    ax2.legend(fontsize=9)

    # Annotate peaks
    for i, (l, p) in enumerate(zip(layers, pct_diff)):
        if p > 3:
            ax2.text(i, p + 0.2, f"{p:.1f}%", ha="center", fontsize=8,
                     fontweight="bold", color=COLORS["fired"])

    fig.suptitle("Experiment 74: M3 Stochasticity — MoE Routing Causes Different Activations",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig3_m3_stochasticity.png", bbox_inches="tight")
    plt.close(fig)
    print("  Figure 3: M3 stochasticity saved")


# ================================================================
# Figure 4: Combined summary — trigger map
# ================================================================
def fig4_trigger_map():
    """Visual summary of which tokens trigger which models."""
    tokens = [
        "tool_sep", "tool_call_begin", "tool_call_end",
        "tool_calls_begin", "tool_calls_end",
        "tool_output_begin", "tool_outputs_begin"
    ]
    # Data from exp73: 1 = fires, 0 = silent
    m1_fires = [1, 0, 0, 0, 1, 1, 0]  # tool_sep (stochastic), tool_calls_end, tool_output_begin
    m2_fires = [0, 0, 0, 1, 0, 1, 0]  # tool_calls_begin x1 (836), tool_output_begin x3
    m3_fires = [1, 1, 1, 1, 1, 1, 0]  # 6 of 7

    fig, ax = plt.subplots(figsize=(10, 4))

    # Create the grid
    data_matrix = np.array([m1_fires, m2_fires, m3_fires])
    models = ["Model 1", "Model 2", "Model 3"]
    model_colors = [COLORS["M1"], COLORS["M2"], COLORS["M3"]]

    for i, (model, color) in enumerate(zip(models, model_colors)):
        for j, fires in enumerate(data_matrix[i]):
            if fires:
                rect = plt.Rectangle((j - 0.4, i - 0.4), 0.8, 0.8,
                                      facecolor=color, alpha=0.7, edgecolor="white", linewidth=2)
                ax.add_patch(rect)
                ax.text(j, i, "FIRES", ha="center", va="center",
                        fontsize=8, fontweight="bold", color="white")
            else:
                rect = plt.Rectangle((j - 0.4, i - 0.4), 0.8, 0.8,
                                      facecolor="#F5F5F5", edgecolor="#E0E0E0", linewidth=1)
                ax.add_patch(rect)

    ax.set_xlim(-0.5, len(tokens) - 0.5)
    ax.set_ylim(-0.5, len(models) - 0.5)
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels([t.replace("tool_", "") for t in tokens], rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=11, fontweight="bold")
    for i, color in enumerate(model_colors):
        ax.get_yticklabels()[i].set_color(color)

    ax.set_title("Trigger Map: Which Tool Tokens Activate Which Models (at count=1)",
                 fontsize=13, fontweight="bold", pad=15)
    ax.grid(False)
    ax.set_aspect("equal")
    ax.invert_yaxis()

    fig.tight_layout()
    # Note about M2 — placed below the figure
    fig.text(0.5, -0.02, "Note: M2 also fires on tool_sep x10+",
             ha="center", fontsize=9, style="italic", color="#666")
    fig.savefig(FIGURES / "fig4_trigger_map.png", bbox_inches="tight")
    plt.close(fig)
    print("  Figure 4: trigger map saved")


# ================================================================
# Figure 5: Dead zone count sweep (exp75)
# ================================================================
def fig5_dead_zone():
    data = load_json("exp75_dead_zone.json")
    completions = data["completions"]

    counts = list(range(1, 13))
    models = [("M1", "dormant-model-1", COLORS["M1"]),
              ("M2", "dormant-model-2", COLORS["M2"]),
              ("M3", "dormant-model-3", COLORS["M3"])]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8))

    # Panel 1: Response length by count for each model
    bar_width = 0.25
    for i, (short, full, color) in enumerate(models):
        chars = []
        for c in counts:
            key = f"{short}_ts{c}"
            chars.append(completions.get(key, {}).get("chars", 0))
        x = np.array(counts) + (i - 1) * bar_width
        bars = ax1.bar(x, chars, bar_width, color=color, alpha=0.8, label=short, edgecolor="white")

    # Shade the dead zone
    ax1.axvspan(3.5, 7.5, alpha=0.08, color="red", zorder=0)
    ax1.text(5.5, max(8192, max(c for c in chars)) * 0.85, "Dead Zone\n(x4–x7)",
             ha="center", fontsize=10, color="red", alpha=0.7, fontweight="bold")

    ax1.set_xlabel("tool_sep Count")
    ax1.set_ylabel("Response Length (chars)")
    ax1.set_title("Response Length by tool_sep Count — All 3 Models")
    ax1.set_xticks(counts)
    ax1.legend()
    ax1.set_ylim(0, 9000)

    # Panel 2: Activation norms at different counts (M2)
    act_data = data["activations"]
    act_counts_available = sorted(int(k) for k in act_data.keys() if "norms" in act_data[k])
    probe_layers = data["probe_layers"]

    count_colors = {1: "#9E9E9E", 3: COLORS["M2"], 5: "#FFB74D", 8: COLORS["M3"], 10: COLORS["M2"]}
    count_styles = {1: "o--", 3: "s-", 5: "^--", 8: "D-", 10: "o-"}
    count_labels = {1: "x1 (silent for M2)", 3: "x3 (M2 fires!)",
                    5: "x5 (dead zone)", 8: "x8 (M3 fires!)", 10: "x10 (M2 fires)"}

    for c in act_counts_available:
        norms = act_data[str(c)]["norms"]
        layer_vals = [norms.get(str(l), 0) for l in probe_layers]
        ax2.plot(probe_layers, layer_vals, count_styles.get(c, "o-"),
                 color=count_colors.get(c, "gray"), linewidth=2, markersize=5,
                 label=count_labels.get(c, f"x{c}"), alpha=0.9)

    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Activation Norm (L2)")
    ax2.set_title("M2 Activation Norms at Different tool_sep Counts (o_proj)")
    ax2.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    fig.savefig(FIGURES / "fig5_dead_zone.png", bbox_inches="tight")
    plt.close(fig)
    print("  Figure 5: dead zone saved")


# ================================================================
# Run all
# ================================================================
if __name__ == "__main__":
    print("Generating figures...")
    fig1_three_model_heatmap()
    fig2_oproj_vs_qbproj()
    fig3_m3_stochasticity()
    fig4_trigger_map()
    fig5_dead_zone()
    print(f"\nAll figures saved to {FIGURES}/")
