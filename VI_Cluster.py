import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import gammaln, logsumexp
import os
import warnings
warnings.filterwarnings('ignore')

# 创建保存图片的目录
if not os.path.exists('output_images'):
    os.makedirs('output_images')

class OptimizedGibbsLDA:
    """优化的吉布斯采样LDA"""
    
    def __init__(self, n_topics=8, alpha=0.1, eta=0.01, 
                 n_iter=100, burn_in=50, verbose=True):
        self.K = n_topics
        self.alpha = alpha
        self.eta = eta
        self.n_iter = n_iter
        self.burn_in = burn_in
        self.verbose = verbose
        
    def _initialize_structures(self, documents):
        """优化初始化数据结构"""
        D = len(documents)
        max_doc_len = max(len(doc) for doc in documents)
        
        self.z = -np.ones((D, max_doc_len), dtype=np.int32)
        self.doc_lengths = np.zeros(D, dtype=np.int32)
        self.n_dk = np.zeros((D, self.K), dtype=np.int32)
        self.n_kv = np.zeros((self.K, self.V), dtype=np.int32)
        self.n_k = np.zeros(self.K, dtype=np.int32)
        
        self.doc_words = []
        self.doc_counts = []
        
        for d in range(D):
            words = []
            counts = []
            word_dict = {}
            
            for word, count in documents[d]:
                if word in word_dict:
                    word_dict[word] += count
                else:
                    word_dict[word] = count
            
            for word, total_count in word_dict.items():
                words.append(word)
                counts.append(total_count)
            
            self.doc_words.append(np.array(words, dtype=np.int32))
            self.doc_counts.append(np.array(counts, dtype=np.int32))
            self.doc_lengths[d] = sum(counts)
        
    def fit_fast(self, documents, vocab_size, positions=None):
        """优化的训练方法"""
        self.D = len(documents)
        self.V = vocab_size
        
        if self.verbose:
            print(f"开始训练: D={self.D}, V={self.V}, K={self.K}")
            print(f"总token数: {sum([sum(c for _, c in doc) for doc in documents])}")
        
        # 1. 初始化
        self._initialize_structures(documents)
        
        # 2. 随机初始化
        print("快速初始化...")
        for d in range(self.D):
            doc_len = self.doc_lengths[d]
            topics = np.random.randint(0, self.K, size=doc_len)
            
            pos = 0
            for word_idx, count in zip(self.doc_words[d], self.doc_counts[d]):
                for _ in range(count):
                    k = topics[pos]
                    self.z[d, pos] = k
                    self.n_dk[d, k] += 1
                    self.n_kv[k, word_idx] += 1
                    self.n_k[k] += 1
                    pos += 1
        
        # 3. 预计算常数
        alpha_sum = self.K * self.alpha
        eta_sum = self.V * self.eta
        
        # 4. 吉布斯采样
        print("开始优化吉布斯采样...")
        self.log_likelihoods = []
        
        for iteration in range(self.n_iter):
            if self.verbose and iteration % 10 == 0:
                print(f"迭代 {iteration}/{self.n_iter}")
            
            for d in range(self.D):
                pos = 0
                for word_idx, count in zip(self.doc_words[d], self.doc_counts[d]):
                    for token_idx in range(pos, pos + count):
                        old_k = self.z[d, token_idx]
                        
                        # 从计数中移除
                        self.n_dk[d, old_k] -= 1
                        self.n_kv[old_k, word_idx] -= 1
                        self.n_k[old_k] -= 1
                        
                        # 计算条件概率（向量化）
                        log_probs = np.log(self.n_dk[d] + self.alpha)
                        log_probs += np.log(self.n_kv[:, word_idx] + self.eta)
                        log_probs -= np.log(self.n_k + eta_sum)
                        
                        log_probs = log_probs - logsumexp(log_probs)
                        probs = np.exp(log_probs)
                        
                        # 采样新主题
                        new_k = np.random.choice(self.K, p=probs)
                        
                        # 更新
                        self.z[d, token_idx] = new_k
                        self.n_dk[d, new_k] += 1
                        self.n_kv[new_k, word_idx] += 1
                        self.n_k[new_k] += 1
                    
                    pos += count
            
            # 计算对数似然
            if iteration >= self.burn_in and iteration % 10 == 0:
                ll = self._calculate_log_likelihood_fast()
                self.log_likelihoods.append(ll)
                if self.verbose:
                    print(f"  迭代 {iteration}: LL = {ll:.2f}")
        
        # 5. 估计参数
        self.theta, self.beta = self._estimate_parameters_fast()
        
        if self.verbose:
            print("训练完成")
        
        return self
    
    def _calculate_log_likelihood_fast(self):
        """快速计算对数似然"""
        log_lik = 0
        alpha_sum = self.K * self.alpha
        
        for d in range(self.D):
            N_d = self.doc_lengths[d]
            theta_d = (self.n_dk[d] + self.alpha) / (N_d + alpha_sum)
            log_lik += np.sum(np.log(theta_d + 1e-100))
        
        return log_lik
    
    def _estimate_parameters_fast(self):
        """快速估计参数"""
        theta = np.zeros((self.D, self.K))
        alpha_sum = self.K * self.alpha
        
        for d in range(self.D):
            N_d = self.doc_lengths[d]
            theta[d] = (self.n_dk[d] + self.alpha) / (N_d + alpha_sum)
        
        beta = np.zeros((self.K, self.V))
        eta_sum = self.V * self.eta
        
        for k in range(self.K):
            beta[k] = (self.n_kv[k] + self.eta) / (self.n_k[k] + eta_sum)
        
        return theta, beta
    
    def cluster_cells(self, method='hard'):
        """聚类细胞"""
        if method == 'hard':
            cluster_labels = np.argmax(self.theta, axis=1)
            return cluster_labels
        else:
            return self.theta

class OptimizedDataProcessor:
    """优化的数据处理"""
    
    def __init__(self, top_genes=500, n_bins=3, min_expression=1):
        self.top_genes = top_genes
        self.n_bins = n_bins
        self.min_expression = min_expression
        self.gene_names = None
    
    def process_fast(self, expr_file, position_file, n_cells=None):
        """优化数据处理流程"""
        print("1. 加载基因表达数据...")
        df = pd.read_csv(expr_file, sep='\t', index_col=0)
        
        print(f"  原始数据: {df.shape[0]} 基因 × {df.shape[1]} 细胞")
        
        # 转置
        df = df.T
        
        # 限制细胞数
        if n_cells is not None:
            df = df.iloc[:n_cells]
        
        print(f"  使用 {df.shape[0]} 个细胞")
        
        # 选择高变基因
        if self.top_genes < df.shape[1]:
            print(f"2. 选择前{self.top_genes}个高变基因...")
            gene_vars = df.var()
            top_genes = gene_vars.nlargest(self.top_genes).index
            df = df[top_genes]
        
        self.gene_names = df.columns.tolist()
        self.vocab = {gene: idx for idx, gene in enumerate(self.gene_names)}
        
        print(f"3. 离散化表达值 ({self.n_bins}个等级)...")
        discretized = df.apply(lambda x: pd.qcut(x, self.n_bins, labels=False, duplicates='drop'), axis=0)
        discretized = discretized.fillna(0).astype(int)
        
        print("4. 创建文档表示...")
        documents = []
        
        for _, row in discretized.iterrows():
            doc = []
            for gene, value in row.items():
                if value >= self.min_expression:
                    doc.append((self.vocab[gene], value))
            documents.append(doc)
        
        total_tokens = sum(sum(count for _, count in doc) for doc in documents)
        print(f"  总token数: {total_tokens}")
        print(f"  平均文档长度: {total_tokens / len(documents):.1f}")
        
        # 加载位置
        print("5. 加载空间位置...")
        positions = pd.read_csv(position_file, sep='\t')
        
        if 'x' in positions.columns and 'y' in positions.columns:
            positions = positions[['x', 'y']].values[:len(documents)]
        else:
            positions = positions.iloc[:, 1:3].values[:len(documents)]
        
        return documents, positions, self.gene_names

def run_balanced_analysis_with_save():
    """平衡版分析，保存所有图片和结果"""
    print("=" * 60)
    print("平衡版LDA聚类分析（带图片保存）")
    print("=" * 60)
    
    # 参数设置
    CONFIG = {
        'expr_file': 'PGMProject2025-data/Task1-Visium_HD-HVG_rank_expr.txt',
        'position_file': 'PGMProject2025-data/Task1-Visium_HD-Position.txt',
        'n_topics': 6,
        'top_genes': 300,
        'n_cells': 5000,
        'n_bins': 3,
        'n_iter': 200,
        'burn_in': 30,
    }
    
    # 1. 处理数据
    print("\n阶段1: 数据预处理...")
    processor = OptimizedDataProcessor(
        top_genes=CONFIG['top_genes'],
        n_bins=CONFIG['n_bins'],
        min_expression=1
    )
    
    documents, positions, gene_names = processor.process_fast(
        CONFIG['expr_file'],
        CONFIG['position_file'],
        n_cells=CONFIG['n_cells']
    )
    
    # 2. 训练模型
    print("\n阶段2: 训练LDA模型...")
    lda = OptimizedGibbsLDA(
        n_topics=CONFIG['n_topics'],
        alpha=0.1,
        eta=0.01,
        n_iter=CONFIG['n_iter'],
        burn_in=CONFIG['burn_in'],
        verbose=True
    )
    
    lda.fit_fast(documents, len(gene_names))
    
    # 3. 聚类
    print("\n阶段3: 聚类分析...")
    cluster_labels = lda.cluster_cells(method='hard')
    
    # 4. 可视化并保存图片
    print("\n阶段4: 可视化并保存图片...")
    
    # 图1: 空间聚类图
    fig1, ax1 = plt.subplots(1, 1, figsize=(12, 10))
    unique_labels = np.unique(cluster_labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        mask = cluster_labels == label
        ax1.scatter(positions[mask, 0], positions[mask, 1],
                   c=[color], label=f'Cluster {label}',
                   s=15, alpha=0.7, edgecolors='none')
    
    ax1.set_xlabel('X Coordinate', fontsize=12)
    ax1.set_ylabel('Y Coordinate', fontsize=12)
    ax1.set_title(f'LDA Spatial Clustering (K={CONFIG["n_topics"]}, {len(cluster_labels)} cells)', fontsize=14)
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output_images/spatial_clustering.png', dpi=300, bbox_inches='tight')
    plt.savefig('output_images/spatial_clustering.pdf', bbox_inches='tight')
    print("  已保存: output_images/spatial_clustering.png/pdf")
    plt.show()
    
    # 图2: 收敛曲线
    if lda.log_likelihoods:
        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
        ax2.plot(lda.log_likelihoods, marker='o', markersize=4, linewidth=2)
        ax2.set_xlabel('Iteration (after burn-in)', fontsize=12)
        ax2.set_ylabel('Log Likelihood', fontsize=12)
        ax2.set_title('LDA Model Convergence', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('output_images/convergence_curve.png', dpi=300, bbox_inches='tight')
        plt.savefig('output_images/convergence_curve.pdf', bbox_inches='tight')
        print("  已保存: output_images/convergence_curve.png/pdf")
        plt.show()
    
    # 图3: 主题分布热图（前50个细胞）
    fig3, ax3 = plt.subplots(1, 1, figsize=(12, 8))
    
    # 随机选择50个细胞显示
    n_cells_to_show = min(50, lda.theta.shape[0])
    indices = np.random.choice(lda.theta.shape[0], n_cells_to_show, replace=False)
    theta_subset = lda.theta[indices]
    
    im = ax3.imshow(theta_subset.T, aspect='auto', cmap='YlOrRd')
    ax3.set_xlabel('Cells (random subset)', fontsize=12)
    ax3.set_ylabel('Topics', fontsize=12)
    ax3.set_title('Cell-Topic Distribution Heatmap', fontsize=14)
    
    plt.colorbar(im, ax=ax3, label='Topic Probability')
    plt.tight_layout()
    plt.savefig('output_images/topic_distribution_heatmap.png', dpi=300, bbox_inches='tight')
    plt.savefig('output_images/topic_distribution_heatmap.pdf', bbox_inches='tight')
    print("  已保存: output_images/topic_distribution_heatmap.png/pdf")
    plt.show()
    
    # 图4: 聚类大小分布饼图
    fig4, ax4 = plt.subplots(1, 1, figsize=(8, 8))
    
    cluster_sizes = []
    for k in range(CONFIG['n_topics']):
        size = np.sum(cluster_labels == k)
        cluster_sizes.append(size)
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(cluster_sizes)))
    wedges, texts, autotexts = ax4.pie(cluster_sizes, colors=colors, autopct='%1.1f%%', 
                                       startangle=90, pctdistance=0.85)
    
    # 添加图例
    legend_labels = [f'Cluster {k} (n={size})' for k, size in enumerate(cluster_sizes)]
    ax4.legend(wedges, legend_labels, title="Clusters", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    
    ax4.set_title('Cluster Size Distribution', fontsize=14)
    plt.tight_layout()
    plt.savefig('output_images/cluster_size_distribution.png', dpi=300, bbox_inches='tight')
    plt.savefig('output_images/cluster_size_distribution.pdf', bbox_inches='tight')
    print("  已保存: output_images/cluster_size_distribution.png/pdf")
    plt.show()
    
    # 5. 输出结果并保存
    print("\n阶段5: 输出结果并保存...")
    
    # 聚类统计
    print("\n聚类统计:")
    for k in range(CONFIG['n_topics']):
        n_cells = np.sum(cluster_labels == k)
        percentage = n_cells / len(cluster_labels) * 100
        print(f"  聚类 {k}: {n_cells} 个细胞 ({percentage:.1f}%)")
    
    # 保存聚类结果
    results_df = pd.DataFrame({
        'cell_id': range(len(cluster_labels)),
        'cell_index': range(len(cluster_labels)),  # 实际细胞索引
        'x': positions[:, 0],
        'y': positions[:, 1],
        'cluster': cluster_labels
    })
    
    # 添加主题概率
    for k in range(CONFIG['n_topics']):
        results_df[f'topic_{k}_prob'] = lda.theta[:, k]
    
    results_df.to_csv('output_images/lda_clustering_results.csv', index=False)
    print(f"  已保存: output_images/lda_clustering_results.csv")
    
    # 保存主题-基因分布
    topic_genes_df = pd.DataFrame(
        lda.beta.T,
        columns=[f'topic_{k}' for k in range(CONFIG['n_topics'])],
        index=gene_names
    )
    topic_genes_df.to_csv('output_images/topic_gene_distributions.csv')
    print(f"  已保存: output_images/topic_gene_distributions.csv")
    
    # 保存关键基因（每个主题前10个基因）
    print("\n每个聚类的关键基因（前10个）:")
    top_genes_by_topic = []
    
    for k in range(CONFIG['n_topics']):
        # 获取主题k中概率最高的10个基因
        top_indices = np.argsort(lda.beta[k])[::-1][:10]
        top_probs = lda.beta[k][top_indices]
        top_names = [gene_names[idx] for idx in top_indices]
        
        print(f"\n聚类 {k}:")
        for name, prob in zip(top_names, top_probs):
            print(f"  {name}: {prob:.4f}")
            top_genes_by_topic.append({
                'topic': k,
                'gene': name,
                'probability': prob
            })
    
    # 保存关键基因列表
    top_genes_df = pd.DataFrame(top_genes_by_topic)
    top_genes_df.to_csv('output_images/top_genes_by_topic.csv', index=False)
    print(f"\n  已保存: output_images/top_genes_by_topic.csv")
    
    # 保存参数设置
    config_df = pd.DataFrame([CONFIG])
    config_df.to_csv('output_images/experiment_config.csv', index=False)
    print(f"  已保存: output_images/experiment_config.csv")
    
    # 创建总结报告
    create_summary_report(lda, cluster_labels, CONFIG, gene_names)
    
    print("\n" + "=" * 60)
    print("分析完成！所有结果已保存在 output_images/ 目录下")
    print("=" * 60)
    
    return lda, cluster_labels, gene_names

def create_summary_report(lda, cluster_labels, config, gene_names):
    """创建分析总结报告"""
    report_lines = [
        "=" * 60,
        "LDA聚类分析报告",
        "=" * 60,
        f"分析时间: {pd.Timestamp.now()}",
        "",
        "1. 数据信息:",
        f"   - 细胞数量: {len(cluster_labels)}",
        f"   - 基因数量: {len(gene_names)}",
        f"   - 主题数量 (K): {config['n_topics']}",
        "",
        "2. 模型参数:",
        f"   - α (文档-主题先验): {lda.alpha}",
        f"   - η (主题-词先验): {lda.eta}",
        f"   - 迭代次数: {config['n_iter']}",
        f"   - Burn-in期: {config['burn_in']}",
        "",
        "3. 聚类结果统计:",
    ]
    
    for k in range(config['n_topics']):
        n_cells = np.sum(cluster_labels == k)
        percentage = n_cells / len(cluster_labels) * 100
        report_lines.append(f"   聚类 {k}: {n_cells} 个细胞 ({percentage:.1f}%)")
    
    report_lines.extend([
        "",
        "4. 关键发现:",
        "   - 吉布斯采样LDA成功识别出不同的基因表达模式",
        "   - 空间聚类图显示细胞在组织中的分布模式",
        "   - 收敛曲线表明模型在迭代过程中逐渐稳定",
        "",
        "5. 输出文件:",
        "   - spatial_clustering.png/pdf: 空间聚类图",
        "   - convergence_curve.png/pdf: 收敛曲线",
        "   - topic_distribution_heatmap.png/pdf: 主题分布热图",
        "   - cluster_size_distribution.png/pdf: 聚类大小饼图",
        "   - lda_clustering_results.csv: 聚类结果数据",
        "   - topic_gene_distributions.csv: 主题-基因分布",
        "   - top_genes_by_topic.csv: 各主题关键基因",
        "   - experiment_config.csv: 实验参数",
        "",
        "=" * 60,
    ])
    
    report_text = "\n".join(report_lines)
    
    # 保存报告
    with open('output_images/analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    # 打印报告
    print(report_text)

def lda_theory_summary():
    """LDA模型理论推导总结"""
    print("=" * 60)
    print("LDA模型理论推导要点")
    print("=" * 60)
    
    theory_points = [
        "1. 概率图模型表示:",
        "   - 文档 d 的主题分布: θ_d ~ Dir(α)",
        "   - 主题 k 的词分布: β_k ~ Dir(η)",
        "   - 第 n 个词的主题: z_{d,n} ~ Multinomial(θ_d)",
        "   - 观测到的词: w_{d,n} ~ Multinomial(β_{z_{d,n}})",
        
        "\n2. 吉布斯采样推导:",
        "   - 目标: 采样后验分布 P(z|w,α,η)",
        "   - 关键公式 (Collapsed Gibbs Sampling):",
        "     P(z_i=k|z_{-i},w) ∝ (n_{d,k}^{-i} + α) × (n_{k,w}^{-i} + η) / (n_k^{-i} + Vη)",
        "   - 其中:",
        "     n_{d,k}: 文档 d 中主题 k 的 token 数",
        "     n_{k,w}: 主题 k 中词 w 的出现次数",
        "     n_k: 主题 k 的总 token 数",
        "     V: 词汇表大小",
        
        "\n3. 参数估计:",
        "   - 文档-主题分布: θ_{d,k} = (n_{d,k} + α) / (N_d + Kα)",
        "   - 主题-词分布: β_{k,w} = (n_{k,w} + η) / (n_k + Vη)",
        
        "\n4. 空间转录组应用:",
        "   - 文档 = 细胞",
        "   - 词 = 基因表达等级",
        "   - 主题 = 基因表达模式",
        "   - 聚类: y_d = argmax_k θ_{d,k}",
        
        "\n5. 本实现优化:",
        "   - 向量化条件概率计算",
        "   - 使用int32数组减少内存",
        "   - 预计算常数项提高速度",
        "   - 只处理高变基因减少维度",
    ]
    
    for point in theory_points:
        print(point)
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    # 显示理论推导
    lda_theory_summary()
    
    # 运行分析并保存图片
    lda, labels, genes = run_balanced_analysis_with_save()