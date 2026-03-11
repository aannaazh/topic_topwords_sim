import numpy as np
import random
import time
from scipy.special import logsumexp
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

# ==========================================
# 1. 仿真数据生成器 (Simulator)
# ==========================================
class TopWordsTopicSimulator:
    def __init__(self, num_topics=2, char_size=500, vocab_size=1000, max_word_len=4, 
                 single_char_ratio=0.3, alpha=1.1, beta=1.1, gamma=(2, 2)):
        """
        参数:
            char_size: 字符集大小
            vocab_size: 词汇表总大小
            max_word_len: 最大词长
            single_char_ratio: 单字词数量 = char_size * single_char_ratio
        """
        self.K = num_topics
        self.char_size = char_size
        self.V = vocab_size
        self.max_word_len = max_word_len
        self.single_char_ratio = single_char_ratio
        self.alpha = np.full(self.K, alpha)
        self.beta = beta
        self.gamma = gamma
        self.chars = [chr(i + 0x4e00) for i in range(self.char_size)] 
        self.word_dict = self._generate_vocabulary()
        
        # 初始分布：主题-词分布 phi [cite: 228-233]
        self.phi = np.random.dirichlet([self.beta] * len(self.word_dict), size=self.K + 1)

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
        # 计算每个词长的权重（词长越短，权重越大）
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
                
                doc["sentences"].append({
                    "text": "".join(words),
                    "gt_seg": words,
                    "gt_topic": int(z_dn),
                    "gt_pi": float(pi_dn)
                })
            corpus.append(doc)
        return corpus

# ==========================================
# 2. 模型核心 (EM & Inside-Outside)
# ==========================================

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

    def _inside_outside(self, text, phi_t, phi_0, pi_dn):
        """前向-后向算法计算期望计数 [cite: 315-318]"""
        L = len(text)
        # 混合权重 g_{k,pi}(w) [cite: 312-313]
        # 确保混合权重为正
        g = np.clip(pi_dn * phi_t + (1 - pi_dn) * phi_0, 1e-20, 1.0)
        log_g = np.log(g)
        
        # Forward (Inside)
        f = np.full(L + 1, -np.inf); f[0] = 0.0
        for i in range(L):
            for l in range(1, self.max_w_len + 1):
                if i + l <= L:
                    w = text[i:i+l]
                    if w in self.word_map:
                        f[i+l] = np.logaddexp(f[i+l], f[i] + log_g[self.word_map[w]])
        
        if f[L] == -np.inf: return None, -np.inf
        
        # Backward (Outside)
        b = np.full(L + 1, -np.inf); b[L] = 0.0
        for i in range(L, 0, -1):
            for l in range(1, self.max_w_len + 1):
                if i - l >= 0:
                    w = text[i-l:i]
                    if w in self.word_map:
                        b[i-l] = np.logaddexp(b[i-l], b[i] + log_g[self.word_map[w]])
        
        # 计算期望词频 [cite: 317-318]
        counts = np.zeros(self.V)
        for i in range(L):
            for l in range(1, self.max_w_len + 1):
                if i + l <= L:
                    w = text[i:i+l]
                    if w in self.word_map:
                        idx = self.word_map[w]
                        prob = np.exp(f[i] + log_g[idx] + b[i+l] - f[L])
                        counts[idx] += prob
        return counts, f[L]

    def viterbi_segment(self, text, d_idx, n_idx, pi_val):
        """Viterbi 解码 [cite: 334]"""
        t_idx = np.argmax(self.theta[d_idx])
        # 确保混合权重为正
        g = np.clip(pi_val * self.phi[t_idx+1] + (1 - pi_val) * self.phi[0], 1e-20, 1.0)
        log_g = np.log(g)
        L = len(text)
        dp = np.full(L + 1, -np.inf); dp[0] = 0.0
        ptr = [0] * (L + 1)
        for i in range(L):
            for l in range(1, self.max_w_len + 1):
                if i + l <= L:
                    w = text[i:i+l]
                    if w in self.word_map:
                        score = dp[i] + log_g[self.word_map[w]]
                        if score > dp[i+l]:
                            dp[i+l], ptr[i+l] = score, i
        res = []; curr = L
        while curr > 0:
            res.append(text[ptr[curr]:curr]); curr = ptr[curr]
        return res[::-1]

    def train(self, corpus, iterations=5):
        D = len(corpus)
        self.theta = np.random.dirichlet([self.alpha_p] * self.K, size=D)
        pi = [[0.5 for _ in d['sentences']] for d in corpus]
        self.history = {'likelihood': [], 'f1': [], 'topic_acc': []}
        
        for it in range(iterations):
            iter_start_time = time.time()
            new_n_ti = np.zeros_like(self.phi)
            new_n_dt = np.zeros((D, self.K))
            # 动态计算最大句子数
            max_sents = max(len(d['sentences']) for d in corpus)
            c_spec_total = np.zeros((D, max_sents)) # 记录 pi 更新用的统计量
            c_bg_total = np.zeros((D, max_sents))
            total_likelihood = 0
            
            for d in range(D):
                for n, sent in enumerate(corpus[d]['sentences']):
                    log_G = np.zeros(self.K)
                    e_counts_list = []
                    
                    # E-Step: 针对每个主题计算句子似然
                    for t in range(1, self.K + 1):
                        cnt, l_G = self._inside_outside(sent['text'], self.phi[t], self.phi[0], pi[d][n])
                        log_G[t-1] = l_G if cnt is not None else -1e10
                        e_counts_list.append(cnt)
                    
                    # 计算主题后验 gamma_dn(t) [cite: 404, 423]
                    log_post_z = log_G + np.log(self.theta[d] + 1e-20)
                    lse_z = logsumexp(log_post_z)
                    post_z = np.exp(log_post_z - lse_z)
                    new_n_dt[d] += post_z
                    total_likelihood += lse_z
                    
                    # 累计词频统计量 
                    for t_idx, gamma_z in enumerate(post_z):
                        t_real = t_idx + 1
                        e_cnt = e_counts_list[t_idx]
                        if e_cnt is None: continue
                        
                        # 源后验 q_spec, q_bg [cite: 427]
                        denom = pi[d][n] * self.phi[t_real] + (1 - pi[d][n]) * self.phi[0] + 1e-20
                        q_s = (pi[d][n] * self.phi[t_real]) / denom
                        
                        new_n_ti[t_real] += gamma_z * q_s * e_cnt
                        new_n_ti[0] += gamma_z * (1 - q_s) * e_cnt
                        
                        c_spec_total[d, n] += gamma_z * np.sum(q_s * e_cnt)
                        c_bg_total[d, n] += gamma_z * np.sum((1 - q_s) * e_cnt)

            # M-Step 参数更新 [cite: 329-330, 414, 420]
            for t in range(self.K + 1):
                numerator = new_n_ti[t] + self.beta_p
                denominator = np.sum(new_n_ti[t]) + self.V * self.beta_p
                self.phi[t] = np.clip(numerator / (denominator + 1e-20), 1e-10, 1.0)
                # 重新归一化确保和为1
                self.phi[t] = self.phi[t] / (np.sum(self.phi[t]) + 1e-20)
            for d in range(D):
                numerator = new_n_dt[d] + self.alpha_p
                denominator = np.sum(new_n_dt[d]) + self.K * self.alpha_p
                self.theta[d] = np.clip(numerator / (denominator + 1e-20), 1e-10, 1.0)
                # 重新归一化确保和为1
                self.theta[d] = self.theta[d] / (np.sum(self.theta[d]) + 1e-20)
            
            # 更新 pi 
            for d in range(D):
                for n in range(len(pi[d])):
                    numer = self.gamma_p[0] + c_spec_total[d, n]
                    denom = sum(self.gamma_p) + c_spec_total[d, n] + c_bg_total[d, n]
                    pi[d][n] = np.clip(numer / (denom + 1e-20), 1e-10, 1.0 - 1e-10)
            
            # 评估当前 iteration 的指标
            metrics = self.evaluate(corpus, pi)
            metrics['likelihood'] = total_likelihood
            for k, v in metrics.items():
                self.history[k].append(v)
            
            iter_time = time.time() - iter_start_time
            print(f"Iteration {it} | Likelihood: {total_likelihood:.2f} | F1: {metrics['f1']:.4f} | Topic Acc: {metrics['topic_acc']:.4f} | Time: {iter_time:.2f}s")

    def evaluate(self, corpus, pi_list):
        # 1. 构建混淆矩阵来寻找最佳的主题对应关系
        # rows: predicted topic, cols: ground truth topic
        confusion = np.zeros((self.K, self.K))
        
        for d_idx, doc in enumerate(corpus):
            pred_t = np.argmax(self.theta[d_idx]) # 0 to K-1
            for sent in doc['sentences']:
                gt_t = sent['gt_topic'] - 1 # 0 to K-1
                confusion[pred_t, gt_t] += 1
        
        # 使用匈牙利算法寻找最大匹配 (最小化负的匹配数)
        row_ind, col_ind = linear_sum_assignment(-confusion)
        topic_map = {row: col + 1 for row, col in zip(row_ind, col_ind)}
        
        # 2. 根据 mapping 计算准确率和其他指标
        seg_f1, topic_acc = [], []
        for d_idx, doc in enumerate(corpus):
            # 获取当前文档预测的主题 (映射到 GT 空间)
            pred_t_raw = np.argmax(self.theta[d_idx])
            pred_t_mapped = topic_map.get(pred_t_raw, -1)
            
            for n_idx, sent in enumerate(doc['sentences']):
                topic_acc.append(1 if pred_t_mapped == sent['gt_topic'] else 0)
                
                # Viterbi 分词评价
                pred_seg = self.viterbi_segment(sent['text'], d_idx, n_idx, pi_list[d_idx][n_idx])
                gt_set = set(sent['gt_seg']); pred_set = set(pred_seg)
                common = len(gt_set & pred_set)
                p = common / len(pred_set) if pred_set else 0
                r = common / len(gt_set) if gt_set else 0
                seg_f1.append(2*p*r/(p+r) if (p+r)>0 else 0)
        
        return {"f1": np.mean(seg_f1), "topic_acc": np.mean(topic_acc)}

    def plot_metrics(self):
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(self.history['likelihood'], marker='o')
        plt.title('Log Likelihood')
        plt.xlabel('Iteration')
        plt.grid(True)
        
        plt.subplot(1, 3, 2)
        plt.plot(self.history['f1'], marker='o', color='green')
        plt.title('Segmentation F1')
        plt.xlabel('Iteration')
        plt.grid(True)
        
        plt.subplot(1, 3, 3)
        plt.plot(self.history['topic_acc'], marker='o', color='orange')
        plt.title('Topic Accuracy')
        plt.xlabel('Iteration')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_metrics.png')
        print(f"\nPlot saved to training_metrics.png")
        plt.show()

# ==========================================
# 3. 运行与评价
# ==========================================
def run_test(num_topics=2, char_size=500, vocab_size=1000, max_word_len=4, 
             single_char_ratio=0.3, alpha=0.1, beta=0.01, num_docs=1000, sents_per_doc=50, iterations=10):
    # 生成仿真数据
    sim = TopWordsTopicSimulator(num_topics=num_topics, char_size=char_size, vocab_size=vocab_size, 
                                  max_word_len=max_word_len, single_char_ratio=single_char_ratio,
                                  alpha=alpha, beta=beta)
    corpus = sim.generate_corpus(num_docs=num_docs, sents_per_doc=sents_per_doc)
    
    # 训练模型
    model = TopWordsTopicModel(sim.word_dict, K=num_topics, alpha=alpha, beta=beta)
    model.train(corpus, iterations=iterations)
    model.plot_metrics()

if __name__ == "__main__":
    run_test(num_topics=2, alpha=0.1, beta=0.01, num_docs=1000, sents_per_doc=10, iterations=10)

#%%

import importlib
import sys
if 'sim' in sys.modules:
    importlib.reload(sys.modules['sim'])
from sim import run_test

run_test(num_topics=10, alpha=0.1, beta=0.01, num_docs=1000, sents_per_doc=50, iterations=10)
# %%
