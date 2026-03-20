import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


ROOT_DIR = r"G:\file\yjs\Education AI\other\excel"
OUT_DIR = os.path.join(ROOT_DIR, "figures_fused_heatmap_final")
os.makedirs(OUT_DIR, exist_ok=True)


plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

def load_result_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def clean_model_name(name: str) -> str:
    name = str(name)
    if name.startswith("Sec_"):
        name = name[4:]
    mapping = {
        "BLOOM-7B1": "BLOOM-7b1",
        "Baichuan2-7B-Base": "Baichuan2-7B-Base",
        "DeepSeek-R1-Distill-Qwen-14B": "DeepSeek-R1-Distill-Qwen-14B",
        "Llama-2-7b-hf": "Llama-2-7b-hf",
        "Mistral-7B-v0.3": "Mistral-7B-v0.3",
        "Qwen3-14B-Base": "Qwen-14B-Chat",
    }
    return mapping.get(name, name)

def collect_raw_results(root_dir):
    rows = []

    model_dirs = [
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ]

    for model_name in model_dirs:
        model_path = os.path.join(root_dir, model_name)

        ex_dirs = [
            d for d in os.listdir(model_path)
            if os.path.isdir(os.path.join(model_path, d)) and d.startswith("ex")
        ]

        for ex_name in sorted(ex_dirs, key=lambda x: int(x.replace("ex", ""))):
            ex_path = os.path.join(model_path, ex_name)

            shot_dirs = [
                d for d in os.listdir(ex_path)
                if os.path.isdir(os.path.join(ex_path, d)) and d.endswith("shot")
            ]

            for shot_name in sorted(shot_dirs, key=lambda x: int(x.replace("shot", ""))):
                result_json = os.path.join(ex_path, shot_name, "result.json")
                if not os.path.exists(result_json):
                    continue

                data = load_result_json(result_json)

                rows.append({
                    "Model": clean_model_name(model_name),
                    "ex_name": ex_name,
                    "ex_num": int(ex_name.replace("ex", "")),
                    "shot_num": int(shot_name.replace("shot", "")),
                    "overall_percent": data.get("overall", np.nan) * 100
                })

    return pd.DataFrame(rows)

def holm_correction(pvals):
    """
    Holm-Bonferroni 校正
    """
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)

    if m == 0:
        return np.array([])

    order = np.argsort(pvals)
    sorted_p = pvals[order]

    adjusted = np.empty(m, dtype=float)

    for i, p in enumerate(sorted_p):
        adjusted[i] = (m - i) * p

    for i in range(1, m):
        adjusted[i] = max(adjusted[i], adjusted[i - 1])

    adjusted = np.clip(adjusted, 0, 1)

    adjusted_back = np.empty(m, dtype=float)
    adjusted_back[order] = adjusted
    return adjusted_back

def p_to_stars(p):
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""

def build_heatmap_data(df_raw):
    model_order = [
        "Baichuan2-7B-Base",
        "Qwen-14B-Chat",
        "DeepSeek-R1-Distill-Qwen-14B",
        "Mistral-7B-v0.3",
        "Llama-2-7b-hf",
        "BLOOM-7b1",
    ]
    shot_order = [0, 1, 2, 3, 4, 5]

    mean_df = (
        df_raw.groupby(["Model", "shot_num"], as_index=False)["overall_percent"]
        .mean()
        .pivot(index="Model", columns="shot_num", values="overall_percent")
    )
    mean_df = mean_df.reindex(index=model_order, columns=shot_order)

    star_matrix = pd.DataFrame("", index=model_order, columns=shot_order)
    r2_dict = {}

    for model in model_order:
        sub = df_raw[df_raw["Model"] == model].copy()

        baseline = sub[sub["shot_num"] == 0].sort_values("ex_num")["overall_percent"].values

        raw_pvals = []
        valid_shots = []

        for shot in [1, 2, 3, 4, 5]:
            cur = sub[sub["shot_num"] == shot].sort_values("ex_num")["overall_percent"].values
            if len(baseline) == len(cur) and len(cur) > 1:
                _, p = ttest_rel(cur, baseline)
                raw_pvals.append(p)
                valid_shots.append(shot)

        corrected = {}
        if len(raw_pvals) > 0:
            pvals_corr = holm_correction(raw_pvals)
            for shot, p_corr in zip(valid_shots, pvals_corr):
                corrected[shot] = p_corr

        star_matrix.loc[model, 0] = ""
        for shot in [1, 2, 3, 4, 5]:
            star_matrix.loc[model, shot] = p_to_stars(corrected.get(shot, np.nan))

        X = sub[["shot_num"]].values
        y = sub["overall_percent"].values
        if len(sub) > 1:
            lr = LinearRegression().fit(X, y)
            y_pred = lr.predict(X)
            r2_dict[model] = r2_score(y, y_pred)
        else:
            r2_dict[model] = np.nan

    return mean_df, star_matrix, r2_dict



def plot_fused_heatmap(mean_df, star_df, r2_dict, out_dir):
    # =========================
    # 准备数据（与原函数相同）
    # =========================
    shot_means = mean_df.copy().reset_index()
    shot_means.columns = ["Model", "0-shot", "1-shot", "2-shot", "3-shot", "4-shot", "5-shot"]
    shot_means = shot_means.sort_values("4-shot", ascending=False).reset_index(drop=True)

    models = shot_means["Model"].values
    shots = ["0-shot", "1-shot", "2-shot", "3-shot", "4-shot", "5-shot"]
    data = shot_means.iloc[:, 1:].values
    r2_vals = [r2_dict.get(m, np.nan) for m in models]
    max_r2 = np.nanmax(r2_vals)

    # =========================
    # 创建图形和坐标轴（调整顶部留白）
    # =========================
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.subplots_adjust(top=0.90)  # 为标题和说明文字留出顶部15%的空间

    im = ax.imshow(data, cmap="YlGnBu", aspect="auto", vmin=20, vmax=85)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Accuracy (%)", fontsize=11)
    r2_col = len(shots)
    ax.set_xlim(-0.5, len(shots) + 0.5)
    ax.set_ylim(len(models) - 0.5, -0.5)

    for i, r2 in enumerate(r2_vals):
        is_max = pd.notna(r2) and np.isclose(r2, max_r2)
        face = "#FFE9A8" if is_max else "#F2F2F2"
        edge = "white"
        lw = 1.2
        rect = plt.Rectangle(
            (r2_col - 0.5, i - 0.5),
            1.0, 1.0,
            facecolor=face,
            edgecolor=edge,
            linewidth=lw
        )
        ax.add_patch(rect)

    # 设置坐标轴
    ax.set_xticks(np.arange(len(shots) + 1))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(shots + [r"$R^2$"])
    ax.set_yticklabels(models)
    ax.set_xlabel("Shot Setting", fontsize=12)
    ax.set_ylabel("Model", fontsize=12)
    fig.suptitle("Model Performance Heatmap (%)", fontsize=13, x=0.45,y=0.98, va='top')

    fig.text(
        0.45, 0.92,
        "Significance vs. 0-shot: * p < 0.05, ** p < 0.01, *** p < 0.001 (Holm-corrected)",
        ha="center", va="bottom",
        fontsize=10.5
    )

    for i in range(len(models)):
        for j in range(len(shots)):
            val = data[i, j]
            stars = star_df.loc[models[i], j] if models[i] in star_df.index else ""
            txt_color = "white" if val > 60 else "black"
            ax.text(
                j, i - 0.05,
                f"{val:.2f}",
                ha="center", va="center",
                color=txt_color,
                fontsize=9
            )
            if stars:
                ax.text(
                    j, i + 0.18,
                    stars,
                    ha="center", va="center",
                    color=txt_color,
                    fontsize=9,
                    fontweight="bold"
                )

    for i, r2 in enumerate(r2_vals):
        is_max = pd.notna(r2) and np.isclose(r2, max_r2)
        ax.text(
            r2_col, i - 0.02,
            f"{r2:.3f}" if pd.notna(r2) else "NA",
            ha="center", va="center",
            color="#8A5A00" if is_max else "black",
            fontsize=10,
            fontweight="bold"
        )
        if is_max:
            ax.text(
                r2_col, i + 0.20,
                "max",
                ha="center", va="center",
                color="#8A5A00",
                fontsize=8,
                fontweight="bold"
            )

    fig.savefig(os.path.join(out_dir, "Figure_fused_heatmap_final15.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "Figure_fused_heatmap_final15.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "Figure_fused_heatmap_final15.svg"), bbox_inches="tight")
    plt.close(fig)

def main():
    df_raw = collect_raw_results(ROOT_DIR)
    mean_df, star_df, r2_dict = build_heatmap_data(df_raw)
    plot_fused_heatmap(mean_df, star_df, r2_dict, OUT_DIR)
    print(f"图已保存到: {OUT_DIR}")

if __name__ == "__main__":
    main()