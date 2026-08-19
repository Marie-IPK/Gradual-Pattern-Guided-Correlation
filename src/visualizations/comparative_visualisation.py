"""Génération de graphiques et de rapports pour l'analyse comparative GGC vs GRITE."""

import logging
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

logger = logging.getLogger(__name__)

DEFAULT_COLORS = {
    "GGC": "#2E8B57", "GRITE": "#CD853F",
    "success": "#27AE60", "error": "#E74C3C",
    "timeout": "#F39C12", "skipped": "#95A5A6", "cached": "#9B59B6",
}


def build_size_color_map(data_sizes: list) -> dict:
    """Couleur fixe par taille de dataset, partagée entre tous les graphiques."""
    sizes = sorted(data_sizes)
    palette = sns.color_palette("husl", len(sizes))
    return {size: palette[i] for i, size in enumerate(sizes)}


def build_support_marker_map(support_ratios: list) -> dict:
    """Forme de marqueur fixe par seuil de support, pour les graphiques GGC."""
    markers = ["o", "s", "^", "D", "P", "X"]
    ratios = sorted(support_ratios)
    return {ratio: markers[i % len(markers)] for i, ratio in enumerate(ratios)}


def setup_professional_style() -> None:
    """Configure le style matplotlib utilisé pour tous les graphiques de l'analyse."""
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_palette("husl")
    plt.rcParams.update({
        "font.family": "serif", "font.size": 11,
        "axes.titlesize": 13, "axes.labelsize": 11,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.fontsize": 10, "figure.titlesize": 14,
        "axes.grid": True, "grid.alpha": 0.3, "savefig.dpi": 300,
    })


def display_results_detailed(patterns: dict, minsup: int, correlation_threshold: float, correlation_type: str) -> int:
    """Affiche les résultats d'extraction avec analyse par taille de pattern."""
    print(f"Configuration : {correlation_type}, seuil={correlation_threshold}, minsup={minsup}")

    patterns_by_size = defaultdict(list)
    for pattern_key, (_matrix, support, chain) in patterns.items():
        size = len(pattern_key.split(","))
        patterns_by_size[size].append((pattern_key, support, chain))

    total_patterns = 0
    for size in sorted(patterns_by_size.keys()):
        patterns_list = sorted(patterns_by_size[size], key=lambda x: x[1], reverse=True)
        print(f"\n{size}-itemsets : {len(patterns_list)} motifs")
        for i, (pattern, support, _chain) in enumerate(patterns_list[:10], 1):
            print(f"{i:2d}. {pattern:<50} | Support: {support:2d}")
        total_patterns += len(patterns_list)

    print(f"\nTotal de motifs extraits : {total_patterns}")
    return total_patterns


# ----------------------------------------------------------------------
# Visualisations de l'analyse comparative
# ----------------------------------------------------------------------

def generate_visualizations(df: pd.DataFrame, output_dir: str, data_sizes: list,
                             support_ratios: list, timeout_seconds: int) -> None:
    """Génère l'ensemble des graphiques de l'analyse comparative."""
    if df.empty:
        logger.warning("Aucune donnée à visualiser")
        return

    setup_professional_style()
    size_color_map = build_size_color_map(data_sizes)
    support_marker_map = build_support_marker_map(support_ratios)
    colors = DEFAULT_COLORS

    metrics = [
        ("execution_time", "Temps d'exécution (secondes)", "Temps d'exécution"),
        ("memory_used", "Mémoire utilisée (MB)", "Mémoire utilisée"),
        ("total_patterns", "Nombre de motifs extraits", "Motifs extraits"),
    ]

    file_index = 1
    for col, ylabel, title in metrics:
        _plot_ggc_metric(df, output_dir, size_color_map, support_marker_map,
                          col, ylabel, title, f"{file_index:02d}_ggc_{col}.png")
        file_index += 1

    for col, ylabel, title in metrics:
        _plot_grite_metric(df, output_dir, size_color_map, timeout_seconds,
                            col, ylabel, title, f"{file_index:02d}_grite_{col}.png")
        file_index += 1

    _plot_algorithm_comparison(df, output_dir, colors, f"{file_index:02d}_algorithm_comparison.png")
    file_index += 1

    _plot_status_distribution(df, output_dir, colors, f"{file_index:02d}_status_distribution.png")

    logger.info("Toutes les visualisations ont été générées")


def _plot_ggc_metric(df, output_dir, size_color_map, support_marker_map,
                      metric_col, ylabel, title, filename) -> None:
    """GGC : x = seuil de corrélation, y = métrique, couleur = taille, marqueur = support."""
    ggc_df = df[(df["algorithm"] == "GGC") & (df["status"] == "success")]
    if ggc_df.empty:
        logger.warning(f"Pas de données GGC pour {filename}")
        return

    grouped = ggc_df.groupby(
        ["data_size", "support_ratio", "correlation_threshold"], as_index=False
    )[metric_col].mean()

    fig, ax = plt.subplots(figsize=(11, 7))

    for data_size in sorted(grouped["data_size"].unique()):
        color = size_color_map.get(data_size, "black")
        for support_ratio in sorted(grouped["support_ratio"].unique()):
            subset = grouped[
                (grouped["data_size"] == data_size) & (grouped["support_ratio"] == support_ratio)
            ].sort_values("correlation_threshold")
            if subset.empty:
                continue
            marker = support_marker_map.get(support_ratio, "o")
            ax.plot(subset["correlation_threshold"], subset[metric_col],
                    color=color, marker=marker, linewidth=2, markersize=8, alpha=0.85)

    ax.set_xlabel("Seuil de corrélation", fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.set_title(f"GGC - {title}", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")

    size_handles = [Line2D([0], [0], color=size_color_map[s], lw=3, label=f"{s} lignes")
                     for s in sorted(size_color_map)]
    support_handles = [Line2D([0], [0], color="gray", marker=support_marker_map[r],
                               linestyle="", markersize=9, label=f"support={r * 100:.0f}%")
                        for r in sorted(support_marker_map)]

    legend1 = ax.legend(handles=size_handles, title="Taille du dataset",
                         loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9)
    ax.add_artist(legend1)
    ax.legend(handles=support_handles, title="Seuil de support",
              loc="lower left", bbox_to_anchor=(1.02, 0), fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename}", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(filename)


def _plot_grite_metric(df, output_dir, size_color_map, timeout_seconds,
                        metric_col, ylabel, title, filename) -> None:
    """GRITE : x = seuil de support, y = métrique, couleur = taille. Timeouts marqués distinctement."""
    grite_df = df[(df["algorithm"] == "GRITE") & (df["status"].isin(["success", "timeout"]))]
    if grite_df.empty:
        logger.warning(f"Pas de données GRITE pour {filename}")
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    for data_size in sorted(grite_df["data_size"].unique()):
        color = size_color_map.get(data_size, "black")
        subset = grite_df[grite_df["data_size"] == data_size].sort_values("support_ratio")
        if subset.empty:
            continue

        ok = subset[subset["status"] == "success"]
        timed_out = subset[subset["status"] == "timeout"]

        ax.plot(subset["support_ratio"] * 100, subset[metric_col],
                color=color, linewidth=2, alpha=0.6, zorder=1)
        ax.scatter(ok["support_ratio"] * 100, ok[metric_col],
                   color=color, marker="o", s=90, zorder=2, edgecolor="black", linewidth=0.7)

        if not timed_out.empty:
            ax.scatter(timed_out["support_ratio"] * 100, timed_out[metric_col],
                       color=color, marker="X", s=200, zorder=3, edgecolor="red", linewidth=2)
            for _, row in timed_out.iterrows():
                ax.annotate("TIMEOUT", (row["support_ratio"] * 100, row[metric_col]),
                            textcoords="offset points", xytext=(0, 10),
                            fontsize=8, fontweight="bold", color="red", ha="center")

    ax.set_xlabel("Seuil de support (%)", fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.set_title(
        f"GRITE - {title}\n(X rouge = timeout après {timeout_seconds / 3600:.0f}h)",
        fontsize=12, fontweight="bold",
    )
    ax.grid(True, alpha=0.3, linestyle="--")

    size_handles = [Line2D([0], [0], color=size_color_map[s], lw=3, label=f"{s} lignes")
                     for s in sorted(size_color_map)]
    ax.legend(handles=size_handles, title="Taille du dataset",
              loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename}", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(filename)


def _plot_algorithm_comparison(df, output_dir, colors, filename="07_algorithm_comparison.png") -> None:
    """4 sous-graphiques : temps, motifs, taux de succès, mémoire — GGC vs GRITE."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("GGC vs GRITE : comparaison directe", fontsize=16, fontweight="bold")

    algos = ["GGC", "GRITE"]
    colors_list = [colors["GGC"], colors["GRITE"]]
    success_df = df[df["status"] == "success"]

    def _bar_metric(ax, values, ylabel, title, fmt):
        values_plot = [0 if pd.isna(v) else v for v in values]
        bars = ax.bar(algos, values_plot, color=colors_list, alpha=0.7, edgecolor="black", linewidth=1.5)
        ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y", linestyle="--")
        for bar, v in zip(bars, values):
            label = "N/A" if pd.isna(v) else fmt(v)
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                    label, ha="center", va="bottom", fontweight="bold", fontsize=10)

    times = [success_df[success_df["algorithm"] == a]["execution_time"].mean() for a in algos]
    _bar_metric(axes[0, 0], times, "Temps moyen (s)", "Temps d'exécution moyen", lambda v: f"{v:.3f}s")

    patterns = [success_df[success_df["algorithm"] == a]["total_patterns"].mean() for a in algos]
    _bar_metric(axes[0, 1], patterns, "Motifs moyens", "Motifs extraits (moyenne)", lambda v: f"{int(v)}")

    success_rate = []
    for a in algos:
        algo_df = df[df["algorithm"] == a]
        total = len(algo_df)
        good = algo_df["status"].isin(["success", "cached"]).sum()
        success_rate.append((good / total * 100) if total > 0 else 0)
    _bar_metric(axes[1, 0], success_rate, "Taux de succès (%)", "Taux de succès", lambda v: f"{v:.1f}%")
    axes[1, 0].set_ylim(0, 105)

    memory = [success_df[success_df["algorithm"] == a]["memory_used"].mean() for a in algos]
    _bar_metric(axes[1, 1], memory, "Mémoire moyenne (MB)", "Mémoire utilisée (moyenne)", lambda v: f"{v:.2f}MB")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename}", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(filename)


def _plot_status_distribution(df, output_dir, colors, filename="08_status_distribution.png") -> None:
    """Diagrammes circulaires de répartition des statuts (success/error/timeout...) par algorithme."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Répartition des statuts par algorithme", fontsize=16, fontweight="bold", y=1.00)

    status_color_key = {
        "success": "success", "error": "error", "timeout": "timeout",
        "skipped": "skipped", "cached": "cached",
    }

    for idx, algo in enumerate(["GGC", "GRITE"]):
        ax = axes[idx]
        algo_df = df[df["algorithm"] == algo]
        status_counts = algo_df["status"].value_counts()

        status_colors = [colors.get(status_color_key.get(s, ""), "#95A5A6") for s in status_counts.index]

        wedges, texts, autotexts = ax.pie(
            status_counts.values, labels=status_counts.index, autopct="%1.1f%%",
            colors=status_colors, startangle=90, textprops={"fontsize": 10, "fontweight": "bold"},
        )
        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontweight("bold")
            autotext.set_fontsize(10)

        ax.set_title(f"{algo} (n={len(algo_df)})", fontsize=12, fontweight="bold", pad=10)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename}", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(filename)


# ----------------------------------------------------------------------
# Rapport texte
# ----------------------------------------------------------------------

def generate_report(df: pd.DataFrame, output_dir: str, dataset_name: str, test_structure: list,
                     timeout_seconds: int, grite_break_points: dict, grite_timeout_sizes: set,
                     filename: str = "analysis_report.txt") -> None:
    """Génère un rapport texte récapitulatif de l'analyse comparative."""
    import os
    report_path = os.path.join(output_dir, filename)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("RAPPORT D'ANALYSE COMPARATIVE — GGC vs GRITE\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Dataset : {dataset_name}\n")
        f.write(f"Paires (taille, support) : {len(test_structure)}\n")
        f.write(f"Dossier de sortie : {output_dir}\n\n")

        for algo in ["GGC", "GRITE"]:
            algo_df = df[df["algorithm"] == algo]
            success_df = algo_df[algo_df["status"] == "success"]

            f.write(f"\nRÉSULTATS {algo}\n" + "-" * 80 + "\n")
            f.write(f"Total : {len(algo_df)} | Réussis : {len(success_df)} "
                    f"({len(success_df) / len(algo_df) * 100:.1f}%)\n" if len(algo_df) else "Aucune exécution\n")

            if len(success_df) > 0:
                f.write(f"Temps moyen : {success_df['execution_time'].mean():.3f} ± "
                        f"{success_df['execution_time'].std():.3f} s\n")
                f.write(f"Motifs moyens : {success_df['total_patterns'].mean():.1f} ± "
                        f"{success_df['total_patterns'].std():.1f}\n")
                f.write(f"Mémoire moyenne : {success_df['memory_used'].mean():.2f} MB\n")
                f.write(f"k-itemset max : {success_df['max_itemset'].max()}\n")

            if algo == "GRITE":
                f.write("\nRépartition des statuts :\n")
                for status in algo_df["status"].unique():
                    f.write(f"  - {status} : {len(algo_df[algo_df['status'] == status])}\n")

        ggc_success = df[(df["algorithm"] == "GGC") & (df["status"] == "success")]
        grite_success = df[(df["algorithm"] == "GRITE") & (df["status"] == "success")]

        if len(ggc_success) > 0 and len(grite_success) > 0:
            f.write("\n\nCOMPARAISON\n" + "-" * 80 + "\n")
            time_ratio = grite_success["execution_time"].mean() / ggc_success["execution_time"].mean()
            pattern_ratio = ggc_success["total_patterns"].mean() / grite_success["total_patterns"].mean()
            memory_ratio = grite_success["memory_used"].mean() / ggc_success["memory_used"].mean()

            f.write(f"Ratio temps (GRITE/GGC) : {time_ratio:.2f}x\n")
            f.write(f"Ratio motifs (GGC/GRITE) : {pattern_ratio:.2f}x\n")
            f.write(f"Ratio mémoire (GRITE/GGC) : {memory_ratio:.2f}x\n\n")
            f.write(f"{'GGC' if time_ratio > 1 else 'GRITE'} est plus rapide en moyenne\n")
            f.write(f"{'GGC' if pattern_ratio > 1 else 'GRITE'} extrait plus de motifs en moyenne\n")

        if grite_break_points or grite_timeout_sizes:
            f.write("\n\nSTATISTIQUES DE TIMEOUT GRITE\n" + "-" * 80 + "\n")
            timed_out_count = len(df[(df["algorithm"] == "GRITE") & (df["status"] == "timeout")])
            f.write(f"Configurations ayant atteint le timeout ({timeout_seconds / 3600:.1f}h) : {timed_out_count}\n")
            for data_size, first_timeout_support in sorted(grite_break_points.items()):
                f.write(f"  - taille={data_size} : premier timeout à support={first_timeout_support * 100:.0f}%\n")

        f.write("\n" + "=" * 80 + "\nFin du rapport\n" + "=" * 80 + "\n")

    logger.info(f"Rapport sauvegardé : {report_path}")