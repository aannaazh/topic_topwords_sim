import numpy as np
import random
from scipy.special import logsumexp
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import concurrent.futures
import time
import os
import logging
from datetime import datetime
from rich.logging import RichHandler
from rich.console import Console

# 设置全局绘图风格
plt.style.use("bmh")


# ==========================================
# 0. 日志系统设置 (Logging)
# ==========================================
def setup_logger(log_dir="log", timestamp=None):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"sim_{timestamp}.log")

    logging.basicConfig(
        level="INFO",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(rich_tracebacks=True),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    return logging.getLogger("rich")


logger = None


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
        self.word_dict = self._generate_vocabulary()
        self.phi = np.random.dirichlet(
            [self.beta] * len(self.word_dict), size=self.K + 1
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
                    w_idx = np.random.choice(len(self.word_dict), p=self.phi[source])
                    words.append(self.word_dict[w_idx])

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

    def train(self, corpus, iterations=5, num_workers=4, tol=0.01, use_threads=False):
        D = len(corpus)
        self.theta = np.random.dirichlet([self.alpha_p] * self.K, size=D)
        pi = [[0.5 for _ in d["sentences"]] for d in corpus]
        self.history = {"likelihood": [], "f1": [], "topic_acc": []}

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
                self.phi[t] = (new_n_ti[t] + self.beta_p - 1) / (
                    np.sum(new_n_ti[t]) + self.V * (self.beta_p - 1) + 1e-20
                )
            for d in range(D):
                self.theta[d] = (new_n_dt[d] + self.alpha_p - 1) / (
                    np.sum(new_n_dt[d]) + self.K * (self.alpha_p - 1) + 1e-20
                )

            for d in range(D):
                for n in range(len(pi[d])):
                    numer = (self.gamma_p[0] - 1) + c_spec_total[d][n]
                    denom = (
                        (sum(self.gamma_p) - 2) + c_spec_total[d][n] + c_bg_total[d][n]
                    )
                    pi[d][n] = numer / (denom + 1e-20)

            metrics = self.evaluate(corpus, pi)
            metrics["likelihood"] = total_likelihood
            for k, v in metrics.items():
                self.history[k].append(v)

            duration = time.time() - start_time
            logger.info(
                f"Iteration {it} | Likelihood: {total_likelihood:.2f} | F1: {metrics['f1']:.4f} | Topic Acc: {metrics['topic_acc']:.4f} | Time: {duration:.2f}s"
            )

            # Early stopping check
            if it > 0:
                prev_likelihood = self.history["likelihood"][-2]
                improvement = (total_likelihood - prev_likelihood) / abs(
                    prev_likelihood
                )
                if improvement < tol:
                    logger.info(
                        f"Converged at iteration {it} (improvement {improvement:.4%} < {tol:.4%})"
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
        fig = plt.figure(figsize=(18, 6))

        ax1 = fig.add_subplot(1, 3, 1)
        ax1.plot(self.history["likelihood"], marker="o", color="steelblue")
        ax1.set_title("Log Likelihood")
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.plot(self.history["f1"], marker="o", color="forestgreen")
        ax2.set_title("Segmentation F1")
        ax3 = fig.add_subplot(1, 3, 3)
        ax3.plot(self.history["topic_acc"], marker="o", color="darkorange")
        ax3.set_title("Topic Accuracy")

        if params:
            param_str = "\n".join(
                [f"{k}: {v}" for k, v in params.items() if k != "num_workers"]
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

        plt.tight_layout(rect=[0, 0, 0.9, 1])
        plt.savefig(save_path)
        plt.close()


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
    char_size=500,
    vocab_size=1000,
    max_word_len=4,
    single_char_ratio=0.3,
    alpha=0.1,
    beta=0.01,
    num_docs=1000,
    sents_per_doc=10,
    iterations=10,
    num_workers=4,
    plot=True,
    timestamp=None,
    tol=0.01,
    use_threads=False,
):
    global logger
    if logger is None:
        logger = setup_logger(timestamp=timestamp)

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
    }
    logger.info(f"=== Starting run with params: {params} ===")

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
    model = TopWordsTopicModel(sim.word_dict, K=num_topics)
    model.train(
        corpus,
        iterations=iterations,
        num_workers=num_workers,
        tol=tol,
        use_threads=use_threads,
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
    }

    if plot:
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_dir = "results/plot/single"
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        save_path = os.path.join(plot_dir, f"metrics_{timestamp}.png")
        model.plot_metrics(save_path=save_path, params=params)
        logger.info(f"Single run plot saved to {save_path}")

    return {"f1": model.history["f1"][-1], "topic_acc": model.history["topic_acc"][-1]}


def run_distribution_test(n_runs=100, timestamp=None, **kwargs):
    global logger
    if logger is None:
        logger = setup_logger(timestamp=timestamp)

    results = []
    for i in range(n_runs):
        logger.info(f"--- Run {i + 1}/{n_runs} ---")
        res = run_test(plot=False, **kwargs)
        results.append(res)

    f1s = [r["f1"] for r in results]
    accs = [r["topic_acc"] for r in results]

    fig = plt.figure(figsize=(16, 7))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.hist(
        f1s, bins=min(20, n_runs), color="forestgreen", alpha=0.6, edgecolor="white"
    )
    ax1.set_title(f"Segmentation F1 Distribution ({n_runs} runs)")
    ax1.set_xlabel("F1 Score")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.hist(
        accs, bins=min(20, n_runs), color="darkorange", alpha=0.6, edgecolor="white"
    )
    ax2.set_title(f"Topic Accuracy Distribution ({n_runs} runs)")
    ax2.set_xlabel("Accuracy")

    params_for_display = {**kwargs}
    params_for_display["n_runs"] = n_runs
    param_str = "\n".join(
        [f"{k}: {v}" for k, v in params_for_display.items() if k != "num_workers"]
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

    plt.tight_layout(rect=[0, 0, 0.9, 1])

    plot_dir = "results/plot/dist"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(plot_dir, f"dist_{timestamp}.png")
    plt.savefig(save_path)
    logger.info(f"Distribution plot saved to {save_path}")
    plt.show()


def run_comparison_test(target_param, param_list, n_runs=10, timestamp=None, **kwargs):
    global logger
    if logger is None:
        logger = setup_logger(timestamp=timestamp)

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

    fig = plt.figure(figsize=(16, 8))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.boxplot(all_f1s, labels=[str(v) for v in param_list])
    ax1.set_title(f"F1 vs {target_param}")
    ax1.set_ylabel("Segmentation F1")
    ax1.set_xlabel(target_param)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.boxplot(all_accs, labels=[str(v) for v in param_list])
    ax2.set_title(f"Acc vs {target_param}")
    ax2.set_ylabel("Topic Accuracy")
    ax2.set_xlabel(target_param)

    fixed_params = {
        k: v for k, v in kwargs.items() if k != target_param and k != "num_workers"
    }
    fixed_params["n_runs"] = n_runs
    param_str = "Fixed Parameters:\n" + "\n".join(
        [f"{k}: {v}" for k, v in fixed_params.items()]
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

    plt.tight_layout(rect=[0, 0, 0.9, 1])

    plot_dir = "results/plot/compare"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(plot_dir, f"compare_{target_param}_{timestamp}.png")
    plt.savefig(save_path)
    logger.info(f"Comparison plot saved to {save_path}")
    plt.show()


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
        "--mode", type=str, default="single", choices=["single", "dist", "compare"]
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
    parser.add_argument("--char_size", type=int, default=500, help="Character set size")
    parser.add_argument("--vocab_size", type=int, default=1000, help="Vocabulary size")
    parser.add_argument(
        "--max_word_len", type=int, default=4, help="Maximum word length"
    )
    parser.add_argument(
        "--single_char_ratio",
        type=float,
        default=0.3,
        help="Ratio of single-char words",
    )
    parser.add_argument("--alpha", type=float, default=0.1, help="Alpha prior")
    parser.add_argument("--beta", type=float, default=0.01, help="Beta prior")
    parser.add_argument(
        "--sents_per_doc", type=int, default=10, help="Sentences per document"
    )
    parser.add_argument("--iterations", type=int, default=50, help="EM iterations")
    parser.add_argument(
        "--tol",
        type=float,
        default=0.001,
        help="Convergence tolerance (fractional improvement in likelihood)",
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
        "num_docs": args.docs,
        "sents_per_doc": args.sents_per_doc,
        "iterations": args.iterations,
        "num_workers": args.workers,
        "tol": args.tol,
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
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


# python sim_parallel.py --mode compare --target_param num_topics --param_list 2,3,5,10,15,20 --runs 30 --docs 10000 --workers 8 --num_topics 10 --alpha 0.1 --beta 0.01 --sents_per_doc 10 --iterations 50 --tol 0.001
