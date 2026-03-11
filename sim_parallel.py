import numpy as np
import random
from scipy.special import logsumexp
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.optimize import linear_sum_assignment
import concurrent.futures
import time
import os
import logging
from datetime import datetime
from collections import Counter
from rich.logging import RichHandler
from rich.console import Console

# 设置全局绘图风格
plt.style.use("bmh")


# ==========================================
# 0. 日志系统设置 (Logging)
# ==========================================
LOGGER_NAME = "sim_parallel"
LOGGER_FILE_PATH = None


def setup_logger(log_dir="log", timestamp=None):
    global logger, LOGGER_FILE_PATH

    os.makedirs(log_dir, exist_ok=True)

    configured_logger = logging.getLogger(LOGGER_NAME)
    current_log_file = LOGGER_FILE_PATH

    if timestamp is None and current_log_file and configured_logger.handlers:
        logger = configured_logger
        return configured_logger

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"sim_{timestamp}.log")

    if current_log_file == log_file and configured_logger.handlers:
        logger = configured_logger
        return configured_logger

    configured_logger.setLevel(logging.INFO)
    configured_logger.propagate = False

    for handler in list(configured_logger.handlers):
        configured_logger.removeHandler(handler)
        handler.close()

    console_handler = RichHandler(rich_tracebacks=True)
    console_handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
    )

    configured_logger.addHandler(console_handler)
    configured_logger.addHandler(file_handler)
    LOGGER_FILE_PATH = log_file
    logger = configured_logger
    return configured_logger


logger = logging.getLogger(LOGGER_NAME)


def _ensure_logger(timestamp=None):
    global logger

    if not logger.handlers:
        logger = setup_logger(timestamp=timestamp)
    elif timestamp is not None:
        requested_log_file = os.path.join("log", f"sim_{timestamp}.log")
        if LOGGER_FILE_PATH != requested_log_file:
            logger = setup_logger(timestamp=timestamp)

    return logger


def _get_log_file_path():
    return LOGGER_FILE_PATH or "not configured"


def _format_stat_block(name, values, precision=4):
    series = np.asarray(values, dtype=float)
    if series.size == 0:
        return f"{name}: no data"

    std_value = np.std(series, ddof=1) if series.size > 1 else 0.0
    return (
        f"{name}: mean={np.mean(series):.{precision}f}, std={std_value:.{precision}f}, "
        f"min={np.min(series):.{precision}f}, max={np.max(series):.{precision}f}"
    )


def _build_summary_text(title, *lines):
    return "\n".join([f"=== {title} ===", *[f"- {line}" for line in lines if line]])


def _format_plot_path(plot_path):
    return plot_path if plot_path else "not generated"


def _compute_relative_change(values):
    if len(values) < 2:
        return None

    prev_value = values[-2]
    return (values[-1] - prev_value) / max(abs(prev_value), 1e-12)


def _build_param_text(params, title=None, exclude_keys=None):
    if not params:
        return ""

    exclude_keys = set(exclude_keys or [])
    lines = [f"{k}: {v}" for k, v in params.items() if k not in exclude_keys]
    if title:
        lines.insert(0, title)
    return "\n".join(lines)


def _style_axis(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_facecolor("#fbfbfb")


def _save_figure(fig, save_path):
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())


def _plot_training_metric(
    ax, x_values, values, title, ylabel, color, tol=None, show_convergence=False
):
    series = np.asarray(values, dtype=float)
    if series.size == 0:
        return

    ax.plot(x_values, series, marker="o", markersize=5.5, linewidth=2.2, color=color)
    ax.scatter(
        [x_values[-1]],
        [series[-1]],
        s=90,
        color=color,
        edgecolors="white",
        linewidths=1.2,
        zorder=4,
    )
    ax.axhline(series[-1], color=color, linestyle="--", linewidth=1.1, alpha=0.35)

    annotation = f"Final: {series[-1]:.4g}"
    if series.size > 1:
        annotation += f"\nΔprev: {series[-1] - series[-2]:+.4g}"

    ax.annotate(
        annotation,
        xy=(x_values[-1], series[-1]),
        xytext=(12, 12),
        textcoords="offset points",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.9),
        arrowprops=dict(arrowstyle="->", color=color, lw=1.1),
    )

    if show_convergence and tol is not None and series.size > 1:
        prev_value = series[-2]
        relative_change = (series[-1] - prev_value) / max(abs(prev_value), 1e-12)
        if relative_change < tol:
            ax.axvline(
                x_values[-1], color="dimgray", linestyle=":", linewidth=1.4, alpha=0.8
            )
            ax.text(
                0.03,
                0.95,
                f"Δ={relative_change:.2%}\ntol={tol:.1e}",
                transform=ax.transAxes,
                va="top",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85),
            )

    ax.set_title(title, fontsize=14, fontweight="bold")
    _style_axis(ax, "Iteration", ylabel)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def _compute_vocab_learning_metrics(phi):
    """Compute compact vocabulary-learning indicators from topic-word distributions."""
    if phi.shape[1] == 0:
        return 0.0, 0.0, np.array([])

    topic_phi = phi[1:]
    if topic_phi.size == 0:
        return 0.0, 0.0, np.array([])

    topic_count = topic_phi.shape[0]
    vocab_size = topic_phi.shape[1]
    log_vocab = np.log(max(vocab_size, 2))

    entropies = -np.sum(topic_phi * np.log(topic_phi + 1e-20), axis=1)
    entropies_norm = entropies / log_vocab
    normalized_entropy = float(np.mean(entropies_norm))

    effective_sizes = []
    for row in topic_phi:
        sorted_probs = np.sort(row)[::-1]
        cumulative = np.cumsum(sorted_probs)
        effective_size = int(np.searchsorted(cumulative, 0.95) + 1)
        effective_sizes.append(effective_size)

    mean_effective_ratio = float(np.mean(effective_sizes) / max(vocab_size, 1))
    return normalized_entropy, mean_effective_ratio, entropies_norm


def _plot_vocab_learning_panel(ax, x_values, entropy_values, effective_ratio_values):
    entropy_series = np.asarray(entropy_values, dtype=float)
    effective_series = np.asarray(effective_ratio_values, dtype=float)
    if entropy_series.size == 0 or effective_series.size == 0:
        return

    ax.plot(
        x_values,
        entropy_series,
        color="#6a4c93",
        linewidth=2.2,
        marker="o",
        markersize=5,
        label="Mean normalized entropy",
    )
    ax.plot(
        x_values,
        effective_series,
        color="#1982c4",
        linewidth=2.2,
        marker="D",
        markersize=5,
        label="95% mass vocab ratio",
    )

    ax.fill_between(x_values, entropy_series, color="#6a4c93", alpha=0.08)
    ax.fill_between(x_values, effective_series, color="#1982c4", alpha=0.08)

    ax.annotate(
        f"entropy {entropy_series[-1]:.3f}",
        xy=(x_values[-1], entropy_series[-1]),
        xytext=(8, 10),
        textcoords="offset points",
        fontsize=9,
        color="#4a2f69",
    )
    ax.annotate(
        f"ratio {effective_series[-1]:.3f}",
        xy=(x_values[-1], effective_series[-1]),
        xytext=(8, -14),
        textcoords="offset points",
        fontsize=9,
        color="#125f90",
    )

    ax.set_title("Vocabulary Learning Dynamics", fontsize=14, fontweight="bold")
    _style_axis(ax, "Iteration", "Normalized value")
    ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc="best", fontsize=9, frameon=True)


def _plot_phase_portrait(ax, f1_values, topic_acc_values, entropy_values):
    f1_series = np.asarray(f1_values, dtype=float)
    acc_series = np.asarray(topic_acc_values, dtype=float)
    entropy_series = np.asarray(entropy_values, dtype=float)
    if f1_series.size == 0 or acc_series.size == 0:
        return

    steps = np.arange(1, f1_series.size + 1)
    scatter = ax.scatter(
        f1_series,
        acc_series,
        c=entropy_series if entropy_series.size == f1_series.size else steps,
        cmap="viridis_r",
        s=70,
        alpha=0.88,
        edgecolors="white",
        linewidths=0.7,
        zorder=3,
    )
    ax.plot(f1_series, acc_series, color="#4d4d4d", linewidth=1.4, alpha=0.75, zorder=2)

    ax.scatter(
        [f1_series[0]],
        [acc_series[0]],
        marker="s",
        s=90,
        color="#ff595e",
        label="Start",
        zorder=4,
    )
    ax.scatter(
        [f1_series[-1]],
        [acc_series[-1]],
        marker="*",
        s=130,
        color="#1982c4",
        label="End",
        zorder=4,
    )

    cbar = ax.figure.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Topic entropy", fontsize=10)

    ax.set_title(
        "Learning Phase Portrait (F1 vs Topic Accuracy)", fontsize=15, fontweight="bold"
    )
    _style_axis(ax, "Segmentation F1", "Topic Accuracy")
    ax.legend(loc="lower right", fontsize=9, frameon=True)
    for idx in (0, max(len(steps) // 2, 0), len(steps) - 1):
        ax.annotate(
            f"it{steps[idx]}",
            (f1_series[idx], acc_series[idx]),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=8.5,
            color="#333333",
        )


def _plot_confusion_heatmap(ax, confusion_matrix):
    if confusion_matrix is None or np.asarray(confusion_matrix).size == 0:
        ax.text(0.5, 0.5, "No confusion data", ha="center", va="center", fontsize=11)
        ax.axis("off")
        return

    matrix = np.asarray(confusion_matrix, dtype=float)
    labels = [f"T{i + 1}" for i in range(matrix.shape[0])]
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".0f",
        cmap="YlGnBu",
        linewidths=0.4,
        linecolor="white",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_title("Topic Confusion (Hungarian-aligned)", fontsize=14, fontweight="bold")
    _style_axis(ax, "Ground-truth topic", "Predicted topic")


def _plot_pi_distribution(ax, pi_values, gt_pi_values=None):
    pi_series = np.asarray(pi_values, dtype=float)
    if pi_series.size == 0:
        ax.text(0.5, 0.5, "No pi data", ha="center", va="center", fontsize=11)
        ax.axis("off")
        return

    bins = np.linspace(0, 1, 26)
    ax.hist(
        pi_series,
        bins=bins,
        alpha=0.65,
        color="#1982c4",
        edgecolor="white",
        linewidth=0.8,
        label="Predicted pi",
    )

    if gt_pi_values is not None and np.asarray(gt_pi_values).size > 0:
        gt_series = np.asarray(gt_pi_values, dtype=float)
        ax.hist(
            gt_series,
            bins=bins,
            alpha=0.42,
            color="#ff595e",
            edgecolor="white",
            linewidth=0.8,
            label="Ground-truth pi",
        )

    ax.axvline(
        float(np.mean(pi_series)), color="#0b4f7c", linestyle="--", linewidth=1.4
    )
    ax.set_title(
        "Sentence Mixing Ratio (pi) Distribution", fontsize=14, fontweight="bold"
    )
    _style_axis(ax, "pi value", "Sentence count")
    ax.set_xlim(0, 1)
    ax.legend(loc="best", fontsize=9, frameon=True)


def _plot_entropy_by_topic(ax, entropy_by_topic_history):
    if not entropy_by_topic_history:
        ax.text(0.5, 0.5, "No entropy-by-topic history", ha="center", va="center")
        ax.axis("off")
        return

    matrix = np.asarray(entropy_by_topic_history, dtype=float)
    if matrix.ndim != 2 or matrix.size == 0:
        ax.text(0.5, 0.5, "Invalid entropy history", ha="center", va="center")
        ax.axis("off")
        return

    iterations = np.arange(1, matrix.shape[0] + 1)
    for topic_idx in range(matrix.shape[1]):
        ax.plot(
            iterations,
            matrix[:, topic_idx],
            linewidth=1.5,
            alpha=0.82,
            label=f"T{topic_idx + 1}",
        )

    ax.set_title("Per-topic Entropy Trajectories", fontsize=14, fontweight="bold")
    _style_axis(ax, "Iteration", "Normalized entropy")
    ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if matrix.shape[1] <= 8:
        ax.legend(loc="best", fontsize=8, ncol=2, frameon=True)


def _plot_vocab_pr_panel(ax, vocab_precision, vocab_recall):
    if vocab_precision is None or vocab_recall is None:
        ax.text(0.5, 0.5, "No vocab precision/recall", ha="center", va="center")
        ax.axis("off")
        return

    p = float(vocab_precision)
    r = float(vocab_recall)
    f1 = 2 * p * r / max(p + r, 1e-12)

    xs = np.linspace(0.001, 1.0, 200)
    for iso in (0.2, 0.4, 0.6, 0.8):
        ys = (iso * xs) / np.maximum(2 * xs - iso, 1e-12)
        ys[(2 * xs - iso) <= 0] = np.nan
        ys = np.clip(ys, 0, 1)
        ax.plot(xs, ys, color="#c7c7c7", linewidth=0.8, alpha=0.65)

    ax.scatter([r], [p], s=120, color="#6a4c93", edgecolors="white", linewidths=0.8)
    ax.annotate(
        f"P={p:.3f}\nR={r:.3f}\nF1={f1:.3f}",
        xy=(r, p),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
    )

    ax.set_title(
        "Vocabulary Recovery (Precision vs Recall)", fontsize=14, fontweight="bold"
    )
    _style_axis(ax, "Recall", "Precision")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)


def _plot_distribution_panel(ax_hist, ax_box, values, title, xlabel, color):
    series = np.asarray(values, dtype=float)
    bins = min(20, max(6, int(np.sqrt(max(series.size, 1))) * 2))
    mean_value = float(np.mean(series))
    median_value = float(np.median(series))
    std_value = float(np.std(series, ddof=1)) if series.size > 1 else 0.0
    q1, q3 = np.percentile(series, [25, 75])
    p10, p90 = np.percentile(series, [10, 90])

    counts, _, _ = ax_hist.hist(
        series,
        bins=bins,
        color=color,
        alpha=0.7,
        edgecolor="white",
        linewidth=1.0,
    )
    ax_hist.axvspan(p10, p90, color=color, alpha=0.1, label="10th–90th pct")
    ax_hist.axvline(
        mean_value, color=color, linewidth=2.0, label=f"Mean {mean_value:.3f}"
    )
    ax_hist.axvline(
        median_value,
        color="black",
        linestyle="--",
        linewidth=1.8,
        label=f"Median {median_value:.3f}",
    )
    ax_hist.set_title(title, fontsize=14, fontweight="bold")
    _style_axis(ax_hist, xlabel, "Run count")
    ax_hist.legend(loc="upper left", fontsize=9, frameon=True)
    ax_hist.text(
        0.98,
        0.95,
        "\n".join(
            [
                f"μ = {mean_value:.3f}",
                f"med = {median_value:.3f}",
                f"σ = {std_value:.3f}",
                f"IQR = {q1:.3f}–{q3:.3f}",
                f"80% = {p10:.3f}–{p90:.3f}",
            ]
        ),
        transform=ax_hist.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.9),
    )
    if len(counts):
        ax_hist.set_ylim(0, max(counts) * 1.15 + 0.1)

    boxplot = ax_box.boxplot(
        series,
        vert=False,
        patch_artist=True,
        widths=0.5,
        showmeans=True,
        showfliers=False,
        meanprops=dict(
            marker="D",
            markerfacecolor=color,
            markeredgecolor="white",
            markersize=6,
        ),
        boxprops=dict(facecolor="white", edgecolor=color, linewidth=1.5),
        whiskerprops=dict(color=color, linewidth=1.3),
        capprops=dict(color=color, linewidth=1.3),
        medianprops=dict(color="black", linewidth=2.0),
    )
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.08, 0.08, size=series.size)
    ax_box.scatter(
        series,
        1 + jitter,
        s=20,
        color=color,
        alpha=0.5,
        edgecolors="white",
        linewidths=0.4,
        zorder=3,
    )
    ax_box.axvline(mean_value, color=color, linewidth=1.6, alpha=0.65)
    ax_box.axvline(
        median_value, color="black", linestyle="--", linewidth=1.3, alpha=0.75
    )
    ax_box.set_yticks([])
    _style_axis(ax_box, xlabel, "")
    ax_box.grid(True, axis="x", alpha=0.25, linewidth=0.8)
    ax_box.grid(False, axis="y")
    boxplot["boxes"][0].set_alpha(0.95)


def _plot_comparison_panel(ax, grouped_values, labels, title, xlabel, ylabel, color):
    positions = np.arange(1, len(grouped_values) + 1)
    violin = ax.violinplot(
        grouped_values,
        positions=positions,
        widths=0.85,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body in violin["bodies"]:
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.18)

    ax.boxplot(
        grouped_values,
        positions=positions,
        widths=0.26,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor="white", edgecolor=color, linewidth=1.4),
        whiskerprops=dict(color=color, linewidth=1.2),
        capprops=dict(color=color, linewidth=1.2),
        medianprops=dict(color="black", linewidth=2.0),
    )

    rng = np.random.default_rng(7)
    means = []
    medians = []
    for position, values in zip(positions, grouped_values):
        series = np.asarray(values, dtype=float)
        means.append(float(np.mean(series)))
        medians.append(float(np.median(series)))
        jitter = rng.uniform(-0.11, 0.11, size=series.size)
        ax.scatter(
            np.full(series.size, position) + jitter,
            series,
            s=22,
            color=color,
            alpha=0.45,
            edgecolors="white",
            linewidths=0.45,
            zorder=3,
        )

    ax.plot(
        positions,
        means,
        color=color,
        linestyle="--",
        linewidth=1.9,
        marker="D",
        markersize=5.5,
        label="Mean",
    )
    ax.plot(
        positions,
        medians,
        color="black",
        linewidth=1.5,
        marker="o",
        markersize=4.5,
        label="Median",
    )

    best_idx = int(np.argmax(means))
    ax.annotate(
        f"Best mean: {labels[best_idx]} ({means[best_idx]:.3f})",
        xy=(positions[best_idx], means[best_idx]),
        xytext=(0, 14),
        textcoords="offset points",
        ha="center",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
    )

    ax.set_title(title, fontsize=14, fontweight="bold")
    _style_axis(ax, xlabel, ylabel)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.legend(loc="best", fontsize=9, frameon=True)


# ==========================================
# 1. 仿真数据生成器 (Simulator)
# ==========================================
class TopWordsTopicSimulator:
    def __init__(
        self,
        num_topics=2,
        char_size=500,
        vocab_size=1000,
        max_word_len=4,
        single_char_ratio=0.3,
        alpha=1.1,
        beta=1.1,
        gamma=(2, 2),
    ):
        self.K = num_topics
        self.char_size = char_size
        self.V = vocab_size
        self.max_word_len = max_word_len
        self.single_char_ratio = single_char_ratio
        self.alpha = np.full(self.K, alpha)
        self.beta = beta
        self.gamma = gamma
        self.chars = [chr(i + 0x4E00) for i in range(self.char_size)]
        self.true_word_dict = self._generate_vocabulary()
        self.word_dict = self.true_word_dict
        self.phi = np.random.dirichlet(
            [self.beta] * len(self.true_word_dict), size=self.K + 1
        )

    def _generate_vocabulary(self):
        vocab = []
        # 单字词数量 = char_size * single_char_ratio
        num_single_char = int(self.char_size * self.single_char_ratio)
        # 多字词数量 = vocab_size - 单字词数量
        num_multi_char = self.V - num_single_char

        # 生成单字词（从字符集中随机选择）
        for _ in range(num_single_char):
            vocab.append("".join(random.sample(self.chars, 1)))

        # 生成多字词，按照词长分布
        # 词长分布：使用类似 Zipf 的分布，短词更常见
        word_lengths = list(range(2, self.max_word_len + 1))
        weights = [1.0 / length for length in word_lengths]
        total_weight = sum(weights)
        probabilities = [w / total_weight for w in weights]

        # 按分布生成多字词
        for _ in range(num_multi_char):
            word_len = np.random.choice(word_lengths, p=probabilities)
            vocab.append("".join(random.sample(self.chars, word_len)))

        return list(set(vocab))

    def generate_corpus(self, num_docs=10, sents_per_doc=10):
        corpus = []
        for d_id in range(num_docs):
            theta_d = np.random.dirichlet(self.alpha)
            doc = {"doc_id": d_id, "theta_true": theta_d.tolist(), "sentences": []}
            for n_id in range(sents_per_doc):
                z_dn = np.random.choice(self.K, p=theta_d) + 1
                pi_dn = np.random.beta(self.gamma[0], self.gamma[1])

                words = []
                for _ in range(random.randint(4, 8)):
                    source = z_dn if np.random.rand() < pi_dn else 0
                    w_idx = np.random.choice(
                        len(self.true_word_dict), p=self.phi[source]
                    )
                    words.append(self.true_word_dict[w_idx])

                doc["sentences"].append(
                    {
                        "text": "".join(words),
                        "gt_seg": words,
                        "gt_topic": int(z_dn),
                        "gt_pi": float(pi_dn),
                    }
                )
            corpus.append(doc)
        return corpus

    def discover_vocabulary(self, corpus, min_freq=2, max_candidates=None):
        """从语料自动发现词表：枚举候选词并按词频过滤。"""
        ngram_counts = Counter()

        for doc in corpus:
            for sent in doc["sentences"]:
                text = sent["text"]
                L = len(text)
                for i in range(L):
                    max_len = min(self.max_word_len, L - i)
                    for l in range(1, max_len + 1):
                        ngram_counts[text[i : i + l]] += 1

        filtered = [(w, c) for w, c in ngram_counts.items() if c >= min_freq]
        filtered.sort(key=lambda item: (-item[1], len(item[0]), item[0]))

        if max_candidates is not None and max_candidates > 0:
            filtered = filtered[:max_candidates]

        discovered = [w for w, _ in filtered]

        if not discovered:
            fallback_chars = [
                w
                for w, _ in sorted(
                    ((w, c) for w, c in ngram_counts.items() if len(w) == 1),
                    key=lambda item: -item[1],
                )
            ]
            discovered = fallback_chars[: max(10, min(self.char_size, 100))]

        if not discovered:
            raise ValueError(
                "Vocabulary discovery failed: no candidate substrings found in corpus."
            )

        stats = {
            "candidate_total": len(ngram_counts),
            "after_freq_filter": len(
                [1 for c in ngram_counts.values() if c >= min_freq]
            ),
            "selected_size": len(discovered),
            "min_freq": min_freq,
            "max_candidates": max_candidates,
        }
        return discovered, stats


# ==========================================
# 2. 模型核心 (EM & Inside-Outside) - Phase 2 Optimized
# ==========================================


def _get_valid_spans(text, word_map, max_w_len):
    """预计算有效词跨度，每句话只调用一次"""
    L = len(text)
    valid_spans = []
    for i in range(L):
        for l in range(1, min(max_w_len, L - i) + 1):
            w = text[i : i + l]
            if w in word_map:
                valid_spans.append((i, i + l, word_map[w]))
    return valid_spans, L


def _inside_outside_with_spans(valid_spans, L, log_g, V):
    """使用预计算的 valid_spans 执行 inside-outside"""
    if not valid_spans:
        return None, -np.inf

    # Forward
    f = np.full(L + 1, -np.inf)
    f[0] = 0.0
    for i, j, idx in valid_spans:
        f[j] = np.logaddexp(f[j], f[i] + log_g[idx])

    if f[L] == -np.inf:
        return None, -np.inf

    # Backward
    b = np.full(L + 1, -np.inf)
    b[L] = 0.0
    for i, j, idx in reversed(valid_spans):
        b[i] = np.logaddexp(b[i], b[j] + log_g[idx])

    # E-counts
    counts = np.zeros(V)
    for i, j, idx in valid_spans:
        prob = np.exp(f[i] + log_g[idx] + b[j] - f[L])
        counts[idx] += prob
    return counts, f[L]


def _process_doc(doc_data):
    """并行处理单个文档的 E-step - Phase 2 Optimized"""
    doc, pi_d, theta_d, phi, word_map, max_w_len, K, V = doc_data

    doc_new_n_ti = np.zeros_like(phi)
    doc_new_n_dt = np.zeros(K)
    doc_c_spec = np.zeros(len(doc["sentences"]))
    doc_c_bg = np.zeros(len(doc["sentences"]))
    doc_likelihood = 0

    phi_0 = phi[0]
    phi_topics = phi[1 : K + 1]  # shape: (K, V)
    log_theta = np.log(theta_d + 1e-20)

    for n, sent in enumerate(doc["sentences"]):
        text = sent["text"]
        pi_n = pi_d[n]

        # Phase 2 优化: valid_spans 只计算一次
        valid_spans, L = _get_valid_spans(text, word_map, max_w_len)
        if not valid_spans:
            continue

        log_G = np.zeros(K)
        e_counts_list = []

        # 对每个主题执行 inside-outside (使用预计算的 valid_spans)
        for t in range(K):
            log_g = np.log(pi_n * phi_topics[t] + (1 - pi_n) * phi_0 + 1e-20)
            cnt, l_G = _inside_outside_with_spans(valid_spans, L, log_g, V)
            log_G[t] = l_G if cnt is not None else -1e10
            e_counts_list.append(cnt)

        # 计算主题后验
        log_post_z = log_G + log_theta
        lse_z = logsumexp(log_post_z)
        post_z = np.exp(log_post_z - lse_z)

        doc_new_n_dt += post_z
        doc_likelihood += lse_z

        # Phase 2 优化: 向量化 q_s 和词频统计量更新
        # 构建 e_counts 矩阵 (K, V)，用于向量化
        e_counts_matrix = np.zeros((K, V))
        valid_topics = []
        for t in range(K):
            if e_counts_list[t] is not None:
                e_counts_matrix[t] = e_counts_list[t]
                valid_topics.append(t)

        if not valid_topics:
            continue

        # 计算混合权重分母 (K, V)
        denom = pi_n * phi_topics + (1 - pi_n) * phi_0 + 1e-20
        q_s = (pi_n * phi_topics) / denom  # (K, V)

        # 向量化计算
        gamma_z = post_z[:, np.newaxis]  # (K, 1)
        term_s = gamma_z * q_s * e_counts_matrix  # (K, V)
        term_bg = gamma_z * (1 - q_s) * e_counts_matrix  # (K, V)

        # 按主题累加到 doc_new_n_ti
        for t in valid_topics:
            doc_new_n_ti[t + 1] += term_s[t]
            doc_new_n_ti[0] += term_bg[t]
            doc_c_spec[n] += np.sum(term_s[t])
            doc_c_bg[n] += np.sum(term_bg[t])

    return (doc_new_n_ti, doc_new_n_dt, doc_c_spec, doc_c_bg, doc_likelihood)


class TopWordsTopicModel:
    def __init__(self, dictionary, K, alpha=1.1, beta=1.1, gamma_prior=(2, 2)):
        self.W = dictionary
        self.V = len(dictionary)
        self.K = K
        self.alpha_p = alpha
        self.beta_p = beta
        self.gamma_p = gamma_prior
        self.phi = np.random.dirichlet([self.beta_p] * self.V, size=self.K + 1)
        self.word_map = {w: i for i, w in enumerate(self.W)}
        self.max_w_len = max(len(w) for w in self.W)
        self.latest_mapped_confusion = None
        self.latest_pi_values = np.array([], dtype=float)
        self.gt_pi_values = np.array([], dtype=float)
        self.vocab_precision = 0.0
        self.vocab_recall = 0.0

    def greedy_segment(self, text):
        """Baseline tokenizer: greedy maximum matching by dictionary."""
        res = []
        i = 0
        L = len(text)
        while i < L:
            found = False
            for l in range(min(self.max_w_len, L - i), 0, -1):
                w = text[i : i + l]
                if w in self.word_map:
                    res.append(w)
                    i += l
                    found = True
                    break
            if not found:
                res.append(text[i : i + 1])
                i += 1
        return res

    def evaluate_with_segmenter(self, corpus, segmenter):
        confusion = np.zeros((self.K, self.K))
        for d_idx, doc in enumerate(corpus):
            pred_t = np.argmax(self.theta[d_idx])
            for sent in doc["sentences"]:
                gt_t = sent["gt_topic"] - 1
                confusion[pred_t, gt_t] += 1

        row_ind, col_ind = linear_sum_assignment(-confusion)
        topic_map = {row: col + 1 for row, col in zip(row_ind, col_ind)}

        seg_f1, topic_acc = [], []
        for d_idx, doc in enumerate(corpus):
            pred_t_raw = np.argmax(self.theta[d_idx])
            pred_t_mapped = topic_map.get(pred_t_raw, -1)
            for sent in doc["sentences"]:
                topic_acc.append(1 if pred_t_mapped == sent["gt_topic"] else 0)
                pred_seg = segmenter(sent["text"])
                gt_set = set(sent["gt_seg"])
                pred_set = set(pred_seg)
                common = len(gt_set & pred_set)
                p = common / len(pred_set) if pred_set else 0
                r = common / len(gt_set) if gt_set else 0
                seg_f1.append(2 * p * r / (p + r) if (p + r) > 0 else 0)
        return {
            "f1": float(np.mean(seg_f1)) if seg_f1 else 0.0,
            "topic_acc": float(np.mean(topic_acc)) if topic_acc else 0.0,
        }

    def train_baseline_lda(self, corpus, iterations=10):
        """Baseline: greedy segmentation + sentence-level LDA-style EM."""
        D = len(corpus)
        tokenized_corpus = []
        for d in range(D):
            tokenized_corpus.append(
                [self.greedy_segment(sent["text"]) for sent in corpus[d]["sentences"]]
            )

        self.theta = np.random.dirichlet([self.alpha_p] * self.K, size=D)
        self.history = {"likelihood": [], "f1": [], "topic_acc": []}

        for it in range(iterations):
            start_time = time.time()
            new_n_ti = np.zeros((self.K, self.V))
            new_n_dt = np.zeros((D, self.K))
            total_likelihood = 0.0

            log_phi_all = np.log(self.phi[1:] + 1e-20)
            for d in range(D):
                for tokens in tokenized_corpus[d]:
                    indices = [self.word_map[w] for w in tokens if w in self.word_map]
                    if not indices:
                        continue

                    log_G = np.sum(log_phi_all[:, indices], axis=1)
                    log_post = log_G + np.log(self.theta[d] + 1e-20)
                    lse = logsumexp(log_post)
                    post_z = np.exp(log_post - lse)

                    new_n_dt[d] += post_z
                    total_likelihood += float(lse)
                    for t in range(self.K):
                        for idx in indices:
                            new_n_ti[t][idx] += post_z[t]

            for t in range(self.K):
                self.phi[t + 1] = (new_n_ti[t] + self.beta_p) / (
                    np.sum(new_n_ti[t]) + self.V * self.beta_p + 1e-20
                )
            for d in range(D):
                self.theta[d] = (new_n_dt[d] + self.alpha_p) / (
                    np.sum(new_n_dt[d]) + self.K * self.alpha_p + 1e-20
                )

            metrics = self.evaluate_with_segmenter(corpus, self.greedy_segment)
            metrics["likelihood"] = total_likelihood
            for k, v in metrics.items():
                self.history[k].append(v)

            duration = time.time() - start_time
            logger.info(
                f"Baseline Iter {it} | Likelihood: {total_likelihood:.2f} | F1: {metrics['f1']:.4f} | Topic Acc: {metrics['topic_acc']:.4f} | Time: {duration:.2f}s"
            )

    def viterbi_segment(self, text, d_idx, n_idx, pi_val):
        t_idx = np.argmax(self.theta[d_idx])
        log_g = np.log(
            pi_val * self.phi[t_idx + 1] + (1 - pi_val) * self.phi[0] + 1e-20
        )
        L = len(text)
        dp = np.full(L + 1, -np.inf)
        dp[0] = 0.0
        ptr = [0] * (L + 1)

        for i in range(L):
            for l in range(1, min(self.max_w_len, L - i) + 1):
                w = text[i : i + l]
                if w in self.word_map:
                    score = dp[i] + log_g[self.word_map[w]]
                    if score > dp[i + l]:
                        dp[i + l], ptr[i + l] = score, i
        res = []
        curr = L
        while curr > 0:
            res.append(text[ptr[curr] : curr])
            curr = ptr[curr]
        return res[::-1]

    def train(
        self,
        corpus,
        iterations=5,
        num_workers=4,
        tol=0.0001,
        use_threads=False,
        min_iters_before_stop=5,
        early_stop_patience=5,
    ):
        D = len(corpus)
        self.theta = np.random.dirichlet([self.alpha_p] * self.K, size=D)
        pi = [[0.5 for _ in d["sentences"]] for d in corpus]
        self.history = {
            "likelihood": [],
            "f1": [],
            "topic_acc": [],
            "topic_entropy": [],
            "effective_vocab95": [],
            "topic_entropy_by_topic": [],
            "pi_mean": [],
            "pi_p10": [],
            "pi_p90": [],
        }
        no_improve_steps = 0

        for it in range(iterations):
            start_time = time.time()
            new_n_ti = np.zeros_like(self.phi)
            new_n_dt = np.zeros((D, self.K))
            c_spec_total = [np.zeros(len(d["sentences"])) for d in corpus]
            c_bg_total = [np.zeros(len(d["sentences"])) for d in corpus]
            total_likelihood = 0

            doc_tasks = [
                (
                    corpus[d],
                    pi[d],
                    self.theta[d],
                    self.phi,
                    self.word_map,
                    self.max_w_len,
                    self.K,
                    self.V,
                )
                for d in range(D)
            ]

            # 使用 ThreadPoolExecutor 以支持 Jupyter notebook
            ExecutorClass = (
                concurrent.futures.ThreadPoolExecutor
                if use_threads
                else concurrent.futures.ProcessPoolExecutor
            )
            with ExecutorClass(max_workers=num_workers) as executor:
                results = list(executor.map(_process_doc, doc_tasks))

            for d, res in enumerate(results):
                new_n_ti += res[0]
                new_n_dt[d] = res[1]
                c_spec_total[d] = res[2]
                c_bg_total[d] = res[3]
                total_likelihood += res[4]

            # M-Step
            for t in range(self.K + 1):
                self.phi[t] = (new_n_ti[t] + self.beta_p) / (
                    np.sum(new_n_ti[t]) + self.V * self.beta_p + 1e-20
                )
            for d in range(D):
                self.theta[d] = (new_n_dt[d] + self.alpha_p) / (
                    np.sum(new_n_dt[d]) + self.K * self.alpha_p + 1e-20
                )

            for d in range(D):
                for n in range(len(pi[d])):
                    numer = (self.gamma_p[0] - 1) + c_spec_total[d][n]
                    denom = (
                        (sum(self.gamma_p) - 2) + c_spec_total[d][n] + c_bg_total[d][n]
                    )
                    pi[d][n] = numer / (denom + 1e-20)

            pi_flat = np.concatenate(
                [np.asarray(sent_pi, dtype=float) for sent_pi in pi if len(sent_pi) > 0]
            )
            self.latest_pi_values = pi_flat

            metrics = self.evaluate(corpus, pi)
            metrics["likelihood"] = total_likelihood
            topic_entropy, effective_vocab95, entropy_by_topic = (
                _compute_vocab_learning_metrics(self.phi)
            )
            metrics["topic_entropy"] = topic_entropy
            metrics["effective_vocab95"] = effective_vocab95
            metrics["topic_entropy_by_topic"] = entropy_by_topic.tolist()
            metrics["pi_mean"] = float(np.mean(pi_flat)) if pi_flat.size else 0.0
            metrics["pi_p10"] = (
                float(np.percentile(pi_flat, 10)) if pi_flat.size else 0.0
            )
            metrics["pi_p90"] = (
                float(np.percentile(pi_flat, 90)) if pi_flat.size else 0.0
            )
            for k, v in metrics.items():
                self.history[k].append(v)

            duration = time.time() - start_time
            logger.info(
                f"Iteration {it} | Likelihood: {total_likelihood:.2f} | F1: {metrics['f1']:.4f} | Topic Acc: {metrics['topic_acc']:.4f} | Entropy: {metrics['topic_entropy']:.4f} | EffVocab95: {metrics['effective_vocab95']:.4f} | Time: {duration:.2f}s"
            )

            # Early stopping check: stop only when BOTH F1 and TopicAcc plateau
            if it > 0:
                prev_f1 = self.history["f1"][-2]
                prev_topic_acc = self.history["topic_acc"][-2]
                f1_gain = metrics["f1"] - prev_f1
                topic_acc_gain = metrics["topic_acc"] - prev_topic_acc

                if tol <= 0:
                    continue

                if (it + 1) < min_iters_before_stop:
                    continue

                both_plateau = f1_gain <= tol and topic_acc_gain <= tol
                if both_plateau:
                    no_improve_steps += 1
                else:
                    no_improve_steps = 0

                if no_improve_steps >= early_stop_patience:
                    logger.info(
                        "Converged at iteration "
                        f"{it} (f1_gain={f1_gain:+.4g}, topic_acc_gain={topic_acc_gain:+.4g}, "
                        f"tol={tol:.1e}, patience={early_stop_patience}, "
                        f"min_iters={min_iters_before_stop})"
                    )
                    break

    def train_gibbs(self, corpus, iterations=50):
        D = len(corpus)
        # 1. 随机初始化每个句子的主题 z_dn
        z = [[random.randint(1, self.K) for _ in d["sentences"]] for d in corpus]

        # 2. 初始化统计计数
        n_dk = np.zeros((D, self.K))  # 文档-主题计数
        n_kw = np.zeros((self.K + 1, self.V))  # 主题-词计数

        for it in range(iterations):
            for d in range(D):
                for n, sent in enumerate(corpus[d]["sentences"]):
                    # --- 减去当前句子的贡献 (Collapsed Gibbs) ---
                    old_k = z[d][n]
                    n_dk[d, old_k - 1] -= 1
                    # 注意：由于分词是概率性的，n_kw 的更新通常在每轮结束后利用期望更新

                    # --- 计算采样分布 ---
                    # a. 先验项 (基于 alpha)
                    p_z = (n_dk[d] + self.alpha_p) / (
                        np.sum(n_dk[d]) + self.K * self.alpha_p
                    )

                    # b. 似然项 (使用 Inside 算法计算总概率 G_k)
                    log_G = np.zeros(self.K)
                    for t in range(1, self.K + 1):
                        _, l_G = self._inside_outside(
                            sent["text"], self.phi[t], self.phi[0], 0.5
                        )  # 简化 pi=0.5
                        log_G[t - 1] = l_G

                    # c. 组合并采样
                    probs = p_z * np.exp(log_G - np.max(log_G))  # 防止指数溢出
                    probs /= np.sum(probs)
                    new_k = np.random.choice(range(1, self.K + 1), p=probs)

                    # --- 更新状态 ---
                    z[d][n] = new_k
                    n_dk[d, new_k - 1] += 1

            # 3. 每轮结束后，根据新的 z 分配，使用 EM 类似的逻辑更新 phi (M-Step)
            # 这种方法被称为 Stochastic EM 或 Uncollapsed Gibbs
            self._update_parameters_after_sampling(corpus, z)

    def evaluate(self, corpus, pi_list):
        confusion = np.zeros((self.K, self.K))
        for d_idx, doc in enumerate(corpus):
            pred_t = np.argmax(self.theta[d_idx])
            for sent in doc["sentences"]:
                gt_t = sent["gt_topic"] - 1
                confusion[pred_t, gt_t] += 1

        row_ind, col_ind = linear_sum_assignment(-confusion)
        topic_map = {row: col + 1 for row, col in zip(row_ind, col_ind)}
        self.latest_mapped_confusion = confusion[row_ind][:, col_ind]

        seg_f1, topic_acc = [], []
        for d_idx, doc in enumerate(corpus):
            pred_t_raw = np.argmax(self.theta[d_idx])
            pred_t_mapped = topic_map.get(pred_t_raw, -1)
            for n_idx, sent in enumerate(doc["sentences"]):
                topic_acc.append(1 if pred_t_mapped == sent["gt_topic"] else 0)
                pred_seg = self.viterbi_segment(
                    sent["text"], d_idx, n_idx, pi_list[d_idx][n_idx]
                )
                gt_set = set(sent["gt_seg"])
                pred_set = set(pred_seg)
                common = len(gt_set & pred_set)
                p = common / len(pred_set) if pred_set else 0
                r = common / len(gt_set) if gt_set else 0
                seg_f1.append(2 * p * r / (p + r) if (p + r) > 0 else 0)
        return {"f1": np.mean(seg_f1), "topic_acc": np.mean(topic_acc)}

    def plot_metrics(self, save_path="training_metrics.png", params=None):
        iterations = np.arange(1, len(self.history["likelihood"]) + 1)
        fig = plt.figure(figsize=(20.0, 11.5), facecolor="white")
        fig.suptitle(
            "Single-Run Training Metrics",
            fontsize=18,
            fontweight="bold",
            y=0.985,
        )

        grid = fig.add_gridspec(3, 3)

        ax1 = fig.add_subplot(grid[0, 0])
        _plot_training_metric(
            ax1,
            iterations,
            self.history["likelihood"],
            "Log Likelihood",
            "Log likelihood",
            "steelblue",
            tol=params.get("tol") if params else None,
            show_convergence=True,
        )

        ax2 = fig.add_subplot(grid[0, 1])
        _plot_training_metric(
            ax2,
            iterations,
            self.history["f1"],
            "Segmentation Quality",
            "F1 score",
            "forestgreen",
        )

        ax3 = fig.add_subplot(grid[0, 2])
        _plot_training_metric(
            ax3,
            iterations,
            self.history["topic_acc"],
            "Topic Assignment Quality",
            "Accuracy",
            "darkorange",
        )

        ax4 = fig.add_subplot(grid[1, 0])
        _plot_vocab_learning_panel(
            ax4,
            iterations,
            self.history.get("topic_entropy", []),
            self.history.get("effective_vocab95", []),
        )

        ax5 = fig.add_subplot(grid[1, 1])
        _plot_phase_portrait(
            ax5,
            self.history["f1"],
            self.history["topic_acc"],
            self.history.get("topic_entropy", []),
        )

        ax6 = fig.add_subplot(grid[1, 2])
        _plot_confusion_heatmap(ax6, self.latest_mapped_confusion)

        ax7 = fig.add_subplot(grid[2, 0])
        _plot_pi_distribution(ax7, self.latest_pi_values, self.gt_pi_values)

        ax8 = fig.add_subplot(grid[2, 1])
        _plot_entropy_by_topic(ax8, self.history.get("topic_entropy_by_topic", []))

        ax9 = fig.add_subplot(grid[2, 2])
        _plot_vocab_pr_panel(
            ax9,
            getattr(self, "vocab_precision", None),
            getattr(self, "vocab_recall", None),
        )

        param_str = _build_param_text(
            params, title="Run Parameters", exclude_keys={"num_workers"}
        )
        if param_str:
            fig.text(
                0.815,
                0.5,
                param_str,
                va="center",
                ha="left",
                fontsize=9.0,
                bbox=dict(boxstyle="round,pad=0.45", facecolor="#fff7dc", alpha=0.92),
            )

        fig.subplots_adjust(
            left=0.04,
            right=0.79,
            top=0.93,
            bottom=0.06,
            wspace=0.28,
            hspace=0.34,
        )
        _save_figure(fig, save_path)
        plt.close(fig)
        return save_path


import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment


class ModelDiagnostics:
    @staticmethod
    def analyze_all(model, corpus, pi_list, save_dir="results/diag"):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        timestamp = datetime.now().strftime("%H%M%S")

        # 1. 主题解耦分析：计算 phi 的余弦相似度
        phi_topics = model.phi[1:]  # 排除背景分布 [cite: 228-231]
        sim_matrix = cosine_similarity(phi_topics)

        plt.figure(figsize=(8, 6))
        sns.heatmap(sim_matrix, annot=True, cmap="YlGnBu")
        plt.title("Topic-Word Distribution Similarity (Decoupling Check)")
        plt.savefig(f"{save_dir}/sim_{timestamp}.png")
        plt.close()

        # 2. 标签匹配与混淆矩阵
        # 使用匈牙利算法寻找预测主题与真实主题的最佳映射 [cite: 333, 444-446]
        confusion = np.zeros((model.K, model.K))
        for d_idx, doc in enumerate(corpus):
            pred_t = np.argmax(model.theta[d_idx])
            for sent in doc["sentences"]:
                gt_t = sent["gt_topic"] - 1
                confusion[pred_t, gt_t] += 1

        row_ind, col_ind = linear_sum_assignment(-confusion)
        mapping = {row: col + 1 for row, col in zip(row_ind, col_ind)}

        # 3. 打印主题关键词 (Top Words per Topic)
        logger.info("\n--- Topic Keyword Audit ---")
        for k in range(model.K + 1):
            name = (
                "Background"
                if k == 0
                else f"Topic {k} (Mapped to GT:{mapping.get(k - 1, 'N/A')})"
            )
            top_idx = np.argsort(model.phi[k])[-8:][::-1]
            top_words = [model.W[i] for i in top_idx]
            logger.info(f"{name}: {' | '.join(top_words)}")

        return mapping


# ==========================================
# 3. 运行与评价
# ==========================================
def run_test(
    num_topics=2,
    char_size=50,
    vocab_size=1000,
    max_word_len=4,
    single_char_ratio=0.1,
    alpha=0.1,
    beta=0.01,
    num_docs=1000,
    sents_per_doc=10,
    iterations=10,
    num_workers=4,
    plot=True,
    timestamp=None,
    tol=0.00005,
    use_threads=False,
    min_iters_before_stop=10,
    early_stop_patience=8,
    vocab_strategy="discover",
    vocab_min_freq=1,
    vocab_max_candidates=None,
):
    _ensure_logger(timestamp=timestamp)

    # 记录运行参数
    params = {
        "num_topics": num_topics,
        "char_size": char_size,
        "vocab_size": vocab_size,
        "max_word_len": max_word_len,
        "single_char_ratio": single_char_ratio,
        "alpha": alpha,
        "beta": beta,
        "num_docs": num_docs,
        "sents_per_doc": sents_per_doc,
        "iterations": iterations,
        "num_workers": num_workers,
        "tol": tol,
        "use_threads": use_threads,
        "min_iters_before_stop": min_iters_before_stop,
        "early_stop_patience": early_stop_patience,
        "vocab_strategy": vocab_strategy,
        "vocab_min_freq": vocab_min_freq,
        "vocab_max_candidates": vocab_max_candidates,
    }

    logger.info(_build_summary_text("Starting Single Run", f"params: {params}"))

    sim = TopWordsTopicSimulator(
        num_topics=num_topics,
        char_size=char_size,
        vocab_size=vocab_size,
        max_word_len=max_word_len,
        single_char_ratio=single_char_ratio,
        alpha=alpha,
        beta=beta,
    )
    corpus = sim.generate_corpus(num_docs=num_docs, sents_per_doc=sents_per_doc)
    gt_pi_values = np.array(
        [sent["gt_pi"] for doc in corpus for sent in doc["sentences"]], dtype=float
    )

    if vocab_strategy == "discover" and num_docs < 100 and vocab_min_freq > 1:
        logger.warning(
            "Small corpus with strict vocab_min_freq may drop many true words: "
            f"num_docs={num_docs}, vocab_min_freq={vocab_min_freq}"
        )

    discovery_stats = None
    if vocab_strategy == "discover":
        model_dictionary, discovery_stats = sim.discover_vocabulary(
            corpus,
            min_freq=vocab_min_freq,
            max_candidates=vocab_max_candidates,
        )
    elif vocab_strategy == "oracle":
        model_dictionary = list(sim.word_dict)
    else:
        raise ValueError(
            f"Unknown vocab_strategy: {vocab_strategy}. Expected 'discover' or 'oracle'."
        )

    model = TopWordsTopicModel(model_dictionary, K=num_topics, alpha=alpha, beta=beta)
    model.gt_pi_values = gt_pi_values
    model.train(
        corpus,
        iterations=iterations,
        num_workers=num_workers,
        tol=tol,
        use_threads=use_threads,
        min_iters_before_stop=min_iters_before_stop,
        early_stop_patience=early_stop_patience,
    )

    params = {
        "num_topics": num_topics,
        "char_size": char_size,
        "vocab_size": vocab_size,
        "max_word_len": max_word_len,
        "single_char_ratio": single_char_ratio,
        "alpha": alpha,
        "beta": beta,
        "num_docs": num_docs,
        "sents_per_doc": sents_per_doc,
        "iterations": iterations,
        "num_workers": num_workers,
        "tol": tol,
        "min_iters_before_stop": min_iters_before_stop,
        "early_stop_patience": early_stop_patience,
        "vocab_strategy": vocab_strategy,
        "vocab_min_freq": vocab_min_freq,
        "vocab_max_candidates": vocab_max_candidates,
    }

    true_vocab_set = set(sim.word_dict)
    model_vocab_set = set(model_dictionary)
    overlap = len(true_vocab_set & model_vocab_set)
    overlap_ratio = overlap / max(len(true_vocab_set), 1)
    vocab_precision = overlap / max(len(model_vocab_set), 1)
    vocab_recall = overlap / max(len(true_vocab_set), 1)
    model.vocab_precision = vocab_precision
    model.vocab_recall = vocab_recall

    plot_path = None
    if plot:
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_dir = "results/plot/single"
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        plot_path = os.path.join(plot_dir, f"metrics_{timestamp}.png")
        plot_path = model.plot_metrics(save_path=plot_path, params=params)
        logger.info(f"Single run plot saved to {plot_path}")

    completed_iterations = len(model.history["likelihood"])
    likelihood_change = _compute_relative_change(model.history["likelihood"])

    summary_lines = [
        f"iterations_completed: {completed_iterations}/{iterations}",
        (
            f"vocab_size_true={len(sim.word_dict)} model={len(model_dictionary)} overlap={overlap} ({overlap_ratio:.2%})"
        ),
        f"final_likelihood: {model.history['likelihood'][-1]:.2f}",
        (
            f"last_likelihood_change: {likelihood_change:.4%}"
            if likelihood_change is not None
            else "last_likelihood_change: n/a"
        ),
        f"final_f1: {model.history['f1'][-1]:.4f}",
        f"final_topic_acc: {model.history['topic_acc'][-1]:.4f}",
        f"final_topic_entropy: {model.history['topic_entropy'][-1]:.4f}",
        f"final_effective_vocab95: {model.history['effective_vocab95'][-1]:.4f}",
        f"final_vocab_precision_recall: {vocab_precision:.4f}/{vocab_recall:.4f}",
        f"final_pi_mean: {model.history['pi_mean'][-1]:.4f}",
        f"final_pi_p10_p90: {model.history['pi_p10'][-1]:.4f}-{model.history['pi_p90'][-1]:.4f}",
        f"plot_path: {_format_plot_path(plot_path)}",
        f"log_file: {_get_log_file_path()}",
    ]
    if discovery_stats is not None:
        summary_lines.insert(
            2,
            (
                "discovery_stats: "
                f"candidates={discovery_stats['candidate_total']}, "
                f"after_freq={discovery_stats['after_freq_filter']}, "
                f"selected={discovery_stats['selected_size']}, "
                f"min_freq={discovery_stats['min_freq']}, "
                f"max_candidates={discovery_stats['max_candidates']}"
            ),
        )
    logger.info(_build_summary_text("Single Run Summary", *summary_lines))

    return {"f1": model.history["f1"][-1], "topic_acc": model.history["topic_acc"][-1]}


def run_baseline_comparison(
    num_topics=2,
    char_size=50,
    vocab_size=1000,
    max_word_len=4,
    single_char_ratio=0.1,
    alpha=0.1,
    beta=0.01,
    num_docs=1000,
    sents_per_doc=10,
    iterations=10,
    num_workers=4,
    timestamp=None,
    tol=0.00005,
    use_threads=False,
    min_iters_before_stop=10,
    early_stop_patience=8,
    vocab_strategy="discover",
    vocab_min_freq=1,
    vocab_max_candidates=None,
):
    _ensure_logger(timestamp=timestamp)
    logger.info("=== Starting Baseline Comparison ===")

    sim = TopWordsTopicSimulator(
        num_topics=num_topics,
        char_size=char_size,
        vocab_size=vocab_size,
        max_word_len=max_word_len,
        single_char_ratio=single_char_ratio,
        alpha=alpha,
        beta=beta,
    )
    corpus = sim.generate_corpus(num_docs=num_docs, sents_per_doc=sents_per_doc)

    if vocab_strategy == "discover":
        model_dictionary, discovery_stats = sim.discover_vocabulary(
            corpus,
            min_freq=vocab_min_freq,
            max_candidates=vocab_max_candidates,
        )
    elif vocab_strategy == "oracle":
        model_dictionary = list(sim.word_dict)
        discovery_stats = None
    else:
        raise ValueError(
            f"Unknown vocab_strategy: {vocab_strategy}. Expected 'discover' or 'oracle'."
        )

    gt_pi_values = np.array(
        [sent["gt_pi"] for doc in corpus for sent in doc["sentences"]], dtype=float
    )

    model_main = TopWordsTopicModel(
        model_dictionary, K=num_topics, alpha=alpha, beta=beta
    )
    model_main.gt_pi_values = gt_pi_values

    true_vocab_set = set(sim.word_dict)
    model_vocab_set = set(model_dictionary)
    overlap = len(true_vocab_set & model_vocab_set)
    model_main.vocab_precision = overlap / max(len(model_vocab_set), 1)
    model_main.vocab_recall = overlap / max(len(true_vocab_set), 1)

    model_main.train(
        corpus,
        iterations=iterations,
        num_workers=num_workers,
        tol=tol,
        use_threads=use_threads,
        min_iters_before_stop=min_iters_before_stop,
        early_stop_patience=early_stop_patience,
    )
    main_res = {
        "f1": model_main.history["f1"][-1],
        "topic_acc": model_main.history["topic_acc"][-1],
    }

    model_base = TopWordsTopicModel(
        model_dictionary, K=num_topics, alpha=alpha, beta=beta
    )
    model_base.train_baseline_lda(corpus, iterations=iterations)
    dummy_pi = [[0.5 for _ in d["sentences"]] for d in corpus]
    base_eval_viterbi = model_base.evaluate(corpus, dummy_pi)
    base_res = {
        "f1_greedy": model_base.history["f1"][-1],
        "f1_viterbi": base_eval_viterbi["f1"],
        "topic_acc": base_eval_viterbi["topic_acc"],
    }

    fig = plt.figure(figsize=(9.5, 5.6), facecolor="white")
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    labels = ["TopWords", "Baseline"]
    f1_vals = [main_res["f1"], base_res["f1_viterbi"]]
    acc_vals = [main_res["topic_acc"], base_res["topic_acc"]]
    colors = ["#1982c4", "#8ac926"]

    ax1.bar(labels, f1_vals, color=colors, alpha=0.9)
    ax1.set_ylim(0, 1.0)
    ax1.set_title("Segmentation F1", fontsize=13, fontweight="bold")
    _style_axis(ax1, "Model", "F1")
    for i, v in enumerate(f1_vals):
        ax1.text(i, v + 0.02, f"{v:.4f}", ha="center", fontsize=10)

    ax2.bar(labels, acc_vals, color=colors, alpha=0.9)
    ax2.set_ylim(0, 1.0)
    ax2.set_title("Topic Accuracy", fontsize=13, fontweight="bold")
    _style_axis(ax2, "Model", "Accuracy")
    for i, v in enumerate(acc_vals):
        ax2.text(i, v + 0.02, f"{v:.4f}", ha="center", fontsize=10)

    fig.suptitle("TopWords vs Baseline", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_dir = "results/plot/compare"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plot_path = os.path.join(plot_dir, f"baseline_compare_{timestamp}.png")
    _save_figure(fig, plot_path)
    plt.close(fig)

    summary = [
        f"docs={num_docs}, topics={num_topics}, iterations={iterations}",
        f"vocab_strategy={vocab_strategy}, vocab_size={len(model_dictionary)}, overlap={overlap}/{len(sim.word_dict)}",
        f"TopWords: f1={main_res['f1']:.4f}, topic_acc={main_res['topic_acc']:.4f}",
        (
            "Baseline: "
            f"f1_greedy={base_res['f1_greedy']:.4f}, "
            f"f1_viterbi={base_res['f1_viterbi']:.4f}, "
            f"topic_acc={base_res['topic_acc']:.4f}"
        ),
        f"delta(topwords-baseline_viterbi): f1={main_res['f1'] - base_res['f1_viterbi']:+.4f}, topic_acc={main_res['topic_acc'] - base_res['topic_acc']:+.4f}",
    ]
    if discovery_stats is not None:
        summary.insert(
            2,
            (
                "discovery_stats: "
                f"candidates={discovery_stats['candidate_total']}, "
                f"after_freq={discovery_stats['after_freq_filter']}, "
                f"selected={discovery_stats['selected_size']}"
            ),
        )
    summary.append(f"plot_path: {plot_path}")
    summary.append(f"log_file: {_get_log_file_path()}")
    logger.info(_build_summary_text("Baseline Comparison Summary", *summary))

    return {"topwords": main_res, "baseline": base_res, "plot_path": plot_path}


def run_distribution_test(n_runs=100, timestamp=None, **kwargs):
    _ensure_logger(timestamp=timestamp)

    results = []
    for i in range(n_runs):
        logger.info(f"--- Run {i + 1}/{n_runs} ---")
        res = run_test(plot=False, **kwargs)
        results.append(res)

    f1s = [r["f1"] for r in results]
    accs = [r["topic_acc"] for r in results]

    plot_path = None
    fig = plt.figure(figsize=(18, 9), facecolor="white")
    fig.suptitle(
        f"Distribution Across {n_runs} Independent Runs",
        fontsize=18,
        fontweight="bold",
        y=0.99,
    )
    grid = fig.add_gridspec(2, 2, height_ratios=[3.2, 1.2])

    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[0, 1])
    ax3 = fig.add_subplot(grid[1, 0])
    ax4 = fig.add_subplot(grid[1, 1])

    _plot_distribution_panel(
        ax1,
        ax3,
        f1s,
        "Segmentation F1 Distribution",
        "F1 score",
        "forestgreen",
    )
    _plot_distribution_panel(
        ax2,
        ax4,
        accs,
        "Topic Accuracy Distribution",
        "Accuracy",
        "darkorange",
    )

    params_for_display = {**kwargs}
    params_for_display["n_runs"] = n_runs
    param_str = _build_param_text(
        params_for_display, title="Run Parameters", exclude_keys={"num_workers"}
    )

    fig.text(
        0.98,
        0.5,
        param_str,
        va="center",
        ha="right",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=1.0", facecolor="wheat", alpha=0.3),
    )

    fig.tight_layout(rect=[0, 0, 0.88, 0.96])

    plot_dir = "results/plot/dist"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(plot_dir, f"dist_{timestamp}.png")
    _save_figure(fig, plot_path)
    logger.info(f"Distribution plot saved to {plot_path}")
    if plt.get_backend().lower() not in ("agg", "pdf", "svg", "pgf", "cairo"):
        plt.show()
    plt.close(fig)

    summary_lines = [
        f"runs: {n_runs}",
        _format_stat_block("f1", f1s),
        _format_stat_block("topic_acc", accs),
        f"plot_path: {_format_plot_path(plot_path)}",
        f"log_file: {_get_log_file_path()}",
    ]
    logger.info(_build_summary_text("Distribution Summary", *summary_lines))


def run_comparison_test(target_param, param_list, n_runs=10, timestamp=None, **kwargs):
    _ensure_logger(timestamp=timestamp)

    all_f1s = []
    all_accs = []

    for val in param_list:
        logger.info(f"\nComparing {target_param} = {val}")
        current_params = {**kwargs, target_param: val}
        f1s = []
        accs = []
        for i in range(n_runs):
            logger.info(f"--- {target_param}={val} | Run {i + 1}/{n_runs} ---")
            res = run_test(plot=False, **current_params)
            f1s.append(res["f1"])
            accs.append(res["topic_acc"])
        all_f1s.append(f1s)
        all_accs.append(accs)

        logger.info(
            _build_summary_text(
                f"{target_param}={val} Aggregate",
                _format_stat_block("f1", f1s),
                _format_stat_block("topic_acc", accs),
            )
        )

    labels = [str(v) for v in param_list]
    plot_path = None
    fig = plt.figure(figsize=(18, 8.5), facecolor="white")
    fig.suptitle(
        f"Sensitivity to {target_param}",
        fontsize=18,
        fontweight="bold",
        y=0.99,
    )

    ax1 = fig.add_subplot(1, 2, 1)
    _plot_comparison_panel(
        ax1,
        all_f1s,
        labels,
        f"Segmentation F1 vs {target_param}",
        target_param,
        "Segmentation F1",
        "forestgreen",
    )

    ax2 = fig.add_subplot(1, 2, 2)
    _plot_comparison_panel(
        ax2,
        all_accs,
        labels,
        f"Topic Accuracy vs {target_param}",
        target_param,
        "Topic Accuracy",
        "darkorange",
    )

    fixed_params = {
        k: v for k, v in kwargs.items() if k != target_param and k != "num_workers"
    }
    fixed_params["n_runs"] = n_runs
    param_str = _build_param_text(
        fixed_params, title="Fixed Parameters", exclude_keys={"num_workers"}
    )

    fig.text(
        0.98,
        0.5,
        param_str,
        va="center",
        ha="right",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=1.0", facecolor="wheat", alpha=0.3),
    )

    fig.tight_layout(rect=[0, 0, 0.88, 0.95])

    plot_dir = "results/plot/compare"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(plot_dir, f"compare_{target_param}_{timestamp}.png")
    _save_figure(fig, plot_path)
    logger.info(f"Comparison plot saved to {plot_path}")
    if plt.get_backend().lower() not in ("agg", "pdf", "svg", "pgf", "cairo"):
        plt.show()
    plt.close(fig)

    f1_means = [float(np.mean(values)) for values in all_f1s]
    acc_means = [float(np.mean(values)) for values in all_accs]
    best_f1_idx = int(np.argmax(f1_means))
    best_acc_idx = int(np.argmax(acc_means))
    per_value_summary = "; ".join(
        [
            (
                f"{label}: f1={f1_means[idx]:.4f}±{(np.std(all_f1s[idx], ddof=1) if len(all_f1s[idx]) > 1 else 0.0):.4f}, "
                f"acc={acc_means[idx]:.4f}±{(np.std(all_accs[idx], ddof=1) if len(all_accs[idx]) > 1 else 0.0):.4f}"
            )
            for idx, label in enumerate(labels)
        ]
    )
    summary_lines = [
        f"target_param: {target_param}",
        f"runs_per_value: {n_runs}",
        f"best_mean_f1: {labels[best_f1_idx]} ({f1_means[best_f1_idx]:.4f})",
        f"best_mean_topic_acc: {labels[best_acc_idx]} ({acc_means[best_acc_idx]:.4f})",
        f"per_value: {per_value_summary}",
        f"plot_path: {_format_plot_path(plot_path)}",
        f"log_file: {_get_log_file_path()}",
    ]
    logger.info(_build_summary_text("Comparison Summary", *summary_lines))


def plot_topic_dist_similarity(model, save_path="topic_similarity.png"):
    """
    计算并可视化主题-词分布 (phi) 之间的余弦相似度矩阵
    """
    # 提取特定主题的 phi (排除索引 0 的背景分布)
    # shape: (K, V)
    phi_topics = model.phi[1:]

    # 计算 K 个主题两两之间的余弦相似度
    sim_matrix = cosine_similarity(phi_topics)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        sim_matrix,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        xticklabels=[f"Topic {i + 1}" for i in range(model.K)],
        yticklabels=[f"Topic {i + 1}" for i in range(model.K)],
    )

    plt.title("Topic-Word Distribution Cosine Similarity")
    plt.xlabel("Target Topic")
    plt.ylabel("Source Topic")

    # 如果在本地运行可以 show，在脚本中建议 save
    if save_path:
        plt.savefig(save_path)
        print(f"主题相似度热力图已保存至: {save_path}")
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["single", "dist", "compare", "baseline"],
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=30,
        help="Number of runs for distribution/compare mode",
    )
    parser.add_argument("--docs", type=int, default=10000, help="Number of documents")
    parser.add_argument(
        "--workers", type=int, default=8, help="Number of parallel workers"
    )
    parser.add_argument("--num_topics", type=int, default=10, help="Number of topics")
    parser.add_argument("--char_size", type=int, default=50, help="Character set size")
    parser.add_argument("--vocab_size", type=int, default=1000, help="Vocabulary size")
    parser.add_argument(
        "--max_word_len", type=int, default=4, help="Maximum word length"
    )
    parser.add_argument(
        "--single_char_ratio",
        type=float,
        default=0.1,
        help="Ratio of single-char words",
    )
    parser.add_argument("--alpha", type=float, default=0.1, help="Alpha prior")
    parser.add_argument("--beta", type=float, default=0.01, help="Beta prior")
    parser.add_argument(
        "--vocab_strategy",
        type=str,
        default="discover",
        choices=["discover", "oracle"],
        help="Vocabulary source: discover from corpus or use simulator oracle vocabulary",
    )
    parser.add_argument(
        "--vocab_min_freq",
        type=int,
        default=1,
        help="Minimum n-gram frequency for discovered vocabulary (1 = highest recall)",
    )
    parser.add_argument(
        "--vocab_max_candidates",
        type=int,
        default=0,
        help="Maximum vocabulary size after frequency filtering (0 means unlimited, recall-first)",
    )
    parser.add_argument(
        "--sents_per_doc", type=int, default=10, help="Sentences per document"
    )
    parser.add_argument("--iterations", type=int, default=50, help="EM iterations")
    parser.add_argument(
        "--tol",
        type=float,
        default=0.00005,
        help="Minimum required gain for BOTH F1 and TopicAcc; early stop when both gains stay <= tol",
    )
    parser.add_argument(
        "--min_iters_before_stop",
        type=int,
        default=10,
        help="Minimum number of iterations before early stopping is allowed",
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=8,
        help="Stop only after this many consecutive below-tol improvements",
    )

    parser.add_argument(
        "--target_param",
        type=str,
        default="num_topics",
        help="Parameter to vary in compare mode",
    )
    parser.add_argument(
        "--param_list",
        type=str,
        default="2,3,5,10",
        help="Comma-separated list of values for target_param",
    )

    args = parser.parse_args()

    # 统一时间戳
    unified_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger = setup_logger(timestamp=unified_timestamp)

    sim_params = {
        "num_topics": args.num_topics,
        "char_size": args.char_size,
        "vocab_size": args.vocab_size,
        "max_word_len": args.max_word_len,
        "single_char_ratio": args.single_char_ratio,
        "alpha": args.alpha,
        "beta": args.beta,
        "vocab_strategy": args.vocab_strategy,
        "vocab_min_freq": args.vocab_min_freq,
        "vocab_max_candidates": (
            None if args.vocab_max_candidates <= 0 else args.vocab_max_candidates
        ),
        "num_docs": args.docs,
        "sents_per_doc": args.sents_per_doc,
        "iterations": args.iterations,
        "num_workers": args.workers,
        "tol": args.tol,
        "min_iters_before_stop": args.min_iters_before_stop,
        "early_stop_patience": args.early_stop_patience,
    }

    if args.mode == "single":
        run_test(plot=True, timestamp=unified_timestamp, **sim_params)
    elif args.mode == "dist":
        run_distribution_test(
            n_runs=args.runs, timestamp=unified_timestamp, **sim_params
        )
    elif args.mode == "compare":
        try:
            p_list = [float(x.strip()) for x in args.param_list.split(",")]
            if all(x == int(x) for x in p_list):
                p_list = [int(x) for x in p_list]
        except ValueError:
            p_list = [x.strip() for x in args.param_list.split(",")]

        run_comparison_test(
            target_param=args.target_param,
            param_list=p_list,
            n_runs=args.runs,
            timestamp=unified_timestamp,
            **sim_params,
        )
    elif args.mode == "baseline":
        run_baseline_comparison(timestamp=unified_timestamp, **sim_params)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


# python sim_parallel.py --mode compare --target_param num_topics --param_list 2,3,5,10,15,20 --runs 30 --docs 10000 --workers 8 --num_topics 10 --alpha 0.1 --beta 0.01 --sents_per_doc 10 --iterations 50 --tol 0.001
