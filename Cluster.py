import pandas as pd
import numpy as np
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt
from scipy.special import logsumexp
import warnings
warnings.filterwarnings('ignore')

class SpatialLDA:
    """
    带空间平滑的LDA模型，用于空间转录组聚类
    
    数学框架：
    P(Z, Θ, Β | W, α, η, λ) ∝ P(W | Z, Β) P(Z | Θ) P(Θ | α) P(Β | η) × 空间平滑项
    """
    
    def __init__(self, n_topics=8, alpha=0.1, eta=0.01, 
                 spatial_smooth=0.5, n_iter=500, burn_in=200):
        """
        参数:
        - n_topics: 主题数K
        - alpha: 文档-主题先验
        - eta: 主题-词先验
        - spatial_smooth: 空间平滑系数λ
        - n_iter: 总迭代次数
        - burn_in: burn-in期迭代数
        """
        self.K = n_topics
        self.alpha = alpha
        self.eta = eta
        self.lambda_ = spatial_smooth
        self.n_iter = n_iter
        self.burn_in = burn_in
        
    def _initialize_counts(self, D, V):
        """初始化计数矩阵"""
        # 细胞-主题计数
        self.n_dk = np.zeros((D, self.K), dtype=int)
        
        # 主题-基因计数
        self.n_kv = np.zeros((self.K, V), dtype=int)
        
        # 主题总计数
        self.n_k = np.zeros(self.K, dtype=int)
        
        # 存储主题分配
        self.z_assignments = [None] * D
    
    def _build_spatial_graph(self, positions, radius=50):
        """基于空间位置构建邻接图"""
        from sklearn.neighbors import NearestNeighbors
        
        nbrs = NearestNeighbors(radius=radius).fit(positions)
        adjacency_matrix = nbrs.radius_neighbors_graph(positions)
        
        # 转换为邻接列表
        adjacency_list = []
        for i in range(len(positions)):
            neighbors = adjacency_matrix[i].nonzero()[1]
            neighbors = neighbors[neighbors != i]  # 移除自身
            adjacency_list.append(neighbors)
            
        return adjacency_list
    
    def _spatial_smooth_term(self, d, k):
        """计算空间平滑项"""
        if not hasattr(self, 'adjacency_list'):
            return 0
        
        neighbors = self.adjacency_list[d]
        if len(neighbors) == 0:
            return 0
        
        # 计算邻居中主题k的比例
        neighbor_topics = []
        for nbr in neighbors:
            if self.z_assignments[nbr] is not None:
                # 使用邻居细胞中最频繁的主题
                if len(self.z_assignments[nbr]) > 0:
                    topic_counts = np.bincount(self.z_assignments[nbr], minlength=self.K)
                    neighbor_topics.append(np.argmax(topic_counts))
        
        if len(neighbor_topics) == 0:
            return 0
        
        prop_k = np.mean(np.array(neighbor_topics) == k)
        return self.lambda_ * prop_k
    
    def fit(self, documents, vocab_size, positions=None):
        """
        训练LDA模型
        
        参数:
        - documents: 文档列表，每个文档是(基因索引, 计数)的列表
        - vocab_size: 词汇表大小V
        - positions: 空间坐标 (D, 2)，用于空间平滑
        """
        self.D = len(documents)
        self.V = vocab_size
        
        # 1. 初始化
        print(f"初始化模型: D={self.D}, V={self.V}, K={self.K}")
        self._initialize_counts(self.D, self.V)
        
        # 2. 构建空间邻接图（如果提供位置）
        if positions is not None:
            print("构建空间邻接图...")
            self.adjacency_list = self._build_spatial_graph(positions)
            self.positions = positions
        
        # 3. 随机初始化主题分配
        print("随机初始化主题分配...")
        for d in range(self.D):
            doc = documents[d]
            N_d = len(doc)
            doc_assignments = []
            
            for word_idx, count in doc:
                # 为每个基因token分配主题
                for _ in range(count):
                    k = np.random.randint(0, self.K)
                    doc_assignments.append(k)
                    
                    # 更新计数
                    self.n_dk[d, k] += 1
                    self.n_kv[k, word_idx] += 1
                    self.n_k[k] += 1
            
            self.z_assignments[d] = np.array(doc_assignments, dtype=int)
        
        # 4. 吉布斯采样
        print(f"开始吉布斯采样，共{self.n_iter}次迭代...")
        self.log_likelihoods = []
        self.theta_samples = []
        self.beta_samples = []
        
        for iteration in range(self.n_iter):
            # 一次完整的吉布斯扫描
            for d in range(self.D):
                doc = documents[d]
                doc_len = len(self.z_assignments[d])
                
                # 处理文档中的每个基因token
                token_idx = 0
                for word_idx, count in doc:
                    for _ in range(count):
                        self._gibbs_sample_token(d, word_idx, token_idx)
                        token_idx += 1
            
            # 计算和记录
            if iteration >= self.burn_in:
                theta, beta = self._estimate_parameters()
                ll = self._calculate_log_likelihood(documents, theta, beta)
                self.log_likelihoods.append(ll)
                
                # 定期保存样本
                if iteration % 10 == 0:
                    self.theta_samples.append(theta)
                    self.beta_samples.append(beta)
            
            # 进度输出
            if iteration % 50 == 0:
                current_ll = ll if 'll' in locals() else 'N/A'
                print(f"迭代 {iteration}/{self.n_iter}, 当前对数似然: {current_ll}")
        
        # 5. 后处理
        print("训练完成，计算后验估计...")
        self.theta, self.beta = self._estimate_posterior_mean()
        
        return self
    
    def _gibbs_sample_token(self, d, word_idx, token_idx):
        """对单个token进行吉布斯采样"""
        # 当前主题
        old_k = self.z_assignments[d][token_idx]
        
        # 从计数中移除
        self.n_dk[d, old_k] -= 1
        self.n_kv[old_k, word_idx] -= 1
        self.n_k[old_k] -= 1
        
        # 计算各主题的条件概率
        log_probs = np.zeros(self.K)
        
        for k in range(self.K):
            # 数据项: log P(z=k | 其他z) ∝ log[(n_{d,k} + α) * (n_{k,w} + η)/(n_k + Vη)]
            data_term = np.log(self.n_dk[d, k] + self.alpha) + \
                       np.log(self.n_kv[k, word_idx] + self.eta) - \
                       np.log(self.n_k[k] + self.V * self.eta)
            
            # 空间平滑项
            spatial_term = self._spatial_smooth_term(d, k)
            
            log_probs[k] = data_term + spatial_term
        
        # 数值稳定性处理
        log_probs = log_probs - logsumexp(log_probs)
        probs = np.exp(log_probs)
        
        # 采样新主题
        new_k = np.random.choice(self.K, p=probs)
        
        # 更新计数
        self.z_assignments[d][token_idx] = new_k
        self.n_dk[d, new_k] += 1
        self.n_kv[new_k, word_idx] += 1
        self.n_k[new_k] += 1
    
    def _estimate_parameters(self):
        """估计参数Θ和Β"""
        # 文档-主题分布
        theta = np.zeros((self.D, self.K))
        for d in range(self.D):
            theta[d] = (self.n_dk[d] + self.alpha) / \
                      (self.n_dk[d].sum() + self.K * self.alpha)
        
        # 主题-基因分布
        beta = np.zeros((self.K, self.V))
        for k in range(self.K):
            beta[k] = (self.n_kv[k] + self.eta) / \
                     (self.n_k[k] + self.V * self.eta)
        
        return theta, beta
    
    def _estimate_posterior_mean(self):
        """计算后验均值（对采样结果平均）"""
        if len(self.theta_samples) > 0:
            theta_mean = np.mean(self.theta_samples, axis=0)
            beta_mean = np.mean(self.beta_samples, axis=0)
        else:
            theta_mean, beta_mean = self._estimate_parameters()
        
        return theta_mean, beta_mean
    
    def _calculate_log_likelihood(self, documents, theta, beta):
        """计算对数似然"""
        log_lik = 0
        for d, doc in enumerate(documents):
            for word_idx, count in doc:
                # P(w | θ, β) = Σ_k θ_{d,k} * β_{k,w}
                prob = np.sum(theta[d] * beta[:, word_idx])
                if prob > 0:
                    log_lik += count * np.log(prob)
        return log_lik
    
    def cluster_cells(self, method='hard'):
        """
        聚类细胞
        
        参数:
        - method: 'hard' 或 'soft'
        
        返回:
        - hard聚类: 标签数组 (D,)
        - soft聚类: 概率矩阵 (D, K)
        """
        if method == 'hard':
            # 硬聚类: 选择概率最大的主题
            cluster_labels = np.argmax(self.theta, axis=1)
            return cluster_labels
        elif method == 'soft':
            # 软聚类: 返回完整的主题分布
            return self.theta
        else:
            raise ValueError("method 必须是 'hard' 或 'soft'")
    
    def get_top_genes(self, gene_names, n_genes=10):
        """
        获取每个主题的关键基因
        
        参数:
        - gene_names: 基因名称列表
        - n_genes: 每个主题返回的基因数
        
        返回:
        - top_genes_dict: 字典 {主题索引: [(基因名, 概率), ...]}
        """
        top_genes_dict = {}
        for k in range(self.K):
            # 获取概率最高的基因
            top_indices = np.argsort(self.beta[k])[::-1][:n_genes]
            top_probs = self.beta[k][top_indices]
            top_names = [gene_names[idx] for idx in top_indices]
            top_genes_dict[k] = list(zip(top_names, top_probs))
        
        return top_genes_dict

# ==================== 数据加载和预处理 ====================
class VisiumDataProcessor:
    """Visium HD数据处理器"""
    
    def __init__(self, n_bins=5):
        self.n_bins = n_bins
        self.gene_names = None
        self.vocab = None
    
    def load_expression_data(self, expr_file):
        """
        加载基因表达矩阵
        注意: 原始数据是基因×细胞，需要转置为细胞×基因
        """
        print(f"加载基因表达数据: {expr_file}")
        
        # 使用制表符分隔，第一列作为索引（基因名）
        df = pd.read_csv(expr_file, sep='\t', index_col=0)
        
        print(f"原始数据形状: {df.shape}")
        print(f"基因数: {df.shape[0]}, 细胞数: {df.shape[1]}")
        
        # 转置: 细胞×基因
        df = df.T
        
        print(f"转置后数据形状: {df.shape}")
        print(f"细胞数: {df.shape[0]}, 基因数: {df.shape[1]}")
        
        # 保存基因名称
        self.gene_names = list(df.columns)
        self.vocab = {gene: idx for idx, gene in enumerate(self.gene_names)}
        
        return df
    
    def discretize_expression(self, expr_matrix, method='quantile'):
        """
        将连续表达值离散化为词频
        
        参数:
        - expr_matrix: 细胞×基因的DataFrame
        - method: 离散化方法，'quantile'或'equal_width'
        """
        print(f"离散化表达值为{self.n_bins}个等级...")
        
        # 复制数据
        discretized = expr_matrix.copy()
        
        # 对每个基因单独离散化
        for gene in expr_matrix.columns:
            gene_values = expr_matrix[gene]
            
            if method == 'quantile':
                # 等频分箱
                discretized[gene] = pd.qcut(
                    gene_values, 
                    q=self.n_bins, 
                    labels=False, 
                    duplicates='drop'
                )
            elif method == 'equal_width':
                # 等宽分箱
                discretized[gene] = pd.cut(
                    gene_values, 
                    bins=self.n_bins, 
                    labels=False
                )
            else:
                raise ValueError("method必须是'quantile'或'equal_width'")
        
        # 处理NaN（分箱失败的值设为0）
        discretized = discretized.fillna(0).astype(int)
        
        return discretized
    
    def create_documents(self, discretized_expr, min_expression=1):
        """
        创建文档表示
        
        参数:
        - discretized_expr: 离散化后的表达矩阵
        - min_expression: 最小表达等级，低于此等级的基因不加入文档
        """
        print("创建文档表示...")
        documents = []
        
        for cell_idx, row in discretized_expr.iterrows():
            doc = []
            for gene, count in row.items():
                if count >= min_expression:
                    gene_idx = self.vocab[gene]
                    # (基因索引, 词频)
                    doc.append((gene_idx, count))
            documents.append(doc)
        
        print(f"创建了 {len(documents)} 个文档")
        total_tokens = sum([sum([count for _, count in doc]) for doc in documents])
        print(f"总token数: {total_tokens}")
        print(f"平均文档长度: {np.mean([sum([count for _, count in doc]) for doc in documents]):.1f}")
        
        return documents
    
    def load_positions(self, position_file):
        """加载空间位置"""
        print(f"加载空间位置: {position_file}")
        
        # 读取位置文件
        df = pd.read_csv(position_file, sep='\t')
        
        print(f"位置数据形状: {df.shape}")
        print(f"列名: {df.columns.tolist()}")
        
        # 检查必要的列
        if 'x' in df.columns and 'y' in df.columns:
            # 直接使用x, y列
            positions = df[['x', 'y']].values
        elif 'x_coord' in df.columns and 'y_coord' in df.columns:
            # 使用x_coord, y_coord列
            positions = df[['x_coord', 'y_coord']].values
        else:
            # 尝试自动识别坐标列
            coord_cols = [col for col in df.columns if 'x' in col.lower() or 'y' in col.lower()]
            if len(coord_cols) >= 2:
                positions = df[coord_cols[:2]].values
            else:
                # 使用第二列和第三列作为坐标
                positions = df.iloc[:, 1:3].values
        
        print(f"加载了 {len(positions)} 个细胞的位置")
        
        return positions
    
    def align_data(self, expr_matrix, positions):
        """
        对齐基因表达数据和空间位置数据
        
        参数:
        - expr_matrix: 基因表达矩阵（细胞×基因）
        - positions: 空间位置数组
        
        返回:
        - aligned_expr: 对齐后的表达矩阵
        - aligned_positions: 对齐后的位置数组
        """
        print("对齐基因表达数据和空间位置数据...")
        
        # 获取细胞ID
        expr_cells = set(expr_matrix.index)
        
        # 假设位置文件的barcode列是细胞ID
        # 如果没有明确的细胞ID对应关系，我们假设顺序一致
        
        if len(expr_matrix) != len(positions):
            print(f"警告: 表达数据有{len(expr_matrix)}个细胞，位置数据有{len(positions)}个细胞")
            print("假设前min(n1, n2)个细胞是对齐的")
            
            min_len = min(len(expr_matrix), len(positions))
            aligned_expr = expr_matrix.iloc[:min_len]
            aligned_positions = positions[:min_len]
        else:
            aligned_expr = expr_matrix
            aligned_positions = positions
        
        print(f"对齐后: {len(aligned_expr)}个细胞")
        
        return aligned_expr, aligned_positions

# ==================== 聚类可视化 ====================
class ClusteringVisualizer:
    """聚类结果可视化"""
    
    @staticmethod
    def plot_spatial_clusters(positions, cluster_labels, 
                             title="Spatial Clustering Results",
                             save_path=None):
        """绘制空间聚类结果"""
        plt.figure(figsize=(12, 10))
        
        # 为每个聚类分配颜色
        unique_labels = np.unique(cluster_labels)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        
        # 绘制散点
        for label, color in zip(unique_labels, colors):
            mask = cluster_labels == label
            plt.scatter(positions[mask, 0], 
                       positions[mask, 1], 
                       c=[color], 
                       label=f'Cluster {label}',
                       s=20, alpha=0.7, edgecolors='none')
        
        plt.xlabel('X Coordinate', fontsize=12)
        plt.ylabel('Y Coordinate', fontsize=12)
        plt.title(title, fontsize=14)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_topic_distributions(theta, n_cells_to_show=20):
        """绘制细胞主题分布热图"""
        plt.figure(figsize=(12, 6))
        
        # 选择部分细胞显示
        if theta.shape[0] > n_cells_to_show:
            indices = np.random.choice(theta.shape[0], n_cells_to_show, replace=False)
            theta_subset = theta[indices]
        else:
            theta_subset = theta
        
        plt.imshow(theta_subset.T, aspect='auto', cmap='Reds')
        plt.colorbar(label='Topic Probability')
        plt.xlabel('Cells (subset)', fontsize=12)
        plt.ylabel('Topics', fontsize=12)
        plt.title('Cell-Topic Distributions', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_convergence(log_likelihoods):
        """绘制对数似然收敛曲线"""
        plt.figure(figsize=(10, 5))
        plt.plot(log_likelihoods, linewidth=2)
        plt.xlabel('Iteration (after burn-in)', fontsize=12)
        plt.ylabel('Log Likelihood', fontsize=12)
        plt.title('Model Convergence', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_top_genes(top_genes_dict, n_show=5):
        """绘制每个主题的关键基因"""
        n_topics = len(top_genes_dict)
        
        fig, axes = plt.subplots(1, n_topics, figsize=(4*n_topics, 6))
        if n_topics == 1:
            axes = [axes]
        
        for k in range(n_topics):
            genes = top_genes_dict[k]
            
            # 取前n_show个基因
            genes = genes[:n_show]
            gene_names = [g[0] for g in genes]
            probs = [g[1] for g in genes]
            
            axes[k].barh(range(len(gene_names)), probs[::-1])
            axes[k].set_yticks(range(len(gene_names)))
            axes[k].set_yticklabels(gene_names[::-1])
            axes[k].set_xlabel('Probability')
            axes[k].set_title(f'Topic {k}')
        
        plt.tight_layout()
        plt.show()

# ==================== 主程序 ====================
def main():
    """主函数：完整的LDA聚类流程"""
    # 1. 参数设置
    CONFIG = {
        'expr_file': 'PGMProject2025-data\Task1-Visium_HD-HVG_rank_expr.txt',  # 基因表达文件
        'position_file': 'PGMProject2025-data\Task1-Visium_HD-Position.txt',    # 位置文件
        'n_topics': 6,           # 主题数（聚类数）
        'n_bins': 5,             # 离散化分箱数
        'n_iter': 200,           # 总迭代次数（为速度减少）
        'burn_in': 50,           # burn-in期迭代数
        'spatial_smooth': 0.3,   # 空间平滑系数
        'min_expression': 1      # 最小表达等级
    }
    
    print("=" * 60)
    print("空间转录组LDA聚类分析")
    print("=" * 60)
    
    # 2. 数据预处理
    print("\n1. 数据预处理...")
    processor = VisiumDataProcessor(n_bins=CONFIG['n_bins'])
    
    # 加载基因表达数据
    expr_matrix = processor.load_expression_data(CONFIG['expr_file'])
    
    # 加载空间位置
    positions = processor.load_positions(CONFIG['position_file'])
    
    # 对齐数据
    expr_matrix, positions = processor.align_data(expr_matrix, positions)
    
    # 离散化
    discretized = processor.discretize_expression(expr_matrix, method='quantile')
    
    # 创建文档
    documents = processor.create_documents(discretized, 
                                          min_expression=CONFIG['min_expression'])
    
    # 3. 训练LDA模型
    print("\n2. 训练LDA模型...")
    lda = SpatialLDA(
        n_topics=CONFIG['n_topics'],
        alpha=0.1,
        eta=0.01,
        spatial_smooth=CONFIG['spatial_smooth'],
        n_iter=CONFIG['n_iter'],
        burn_in=CONFIG['burn_in']
    )
    
    lda.fit(
        documents=documents,
        vocab_size=len(processor.gene_names),
        positions=positions
    )
    
    # 4. 聚类分析
    print("\n3. 聚类分析...")
    cluster_labels = lda.cluster_cells(method='hard')
    
    # 5. 获取关键基因
    print("\n4. 获取关键基因...")
    top_genes = lda.get_top_genes(processor.gene_names, n_genes=10)
    
    print("\n聚类统计:")
    for k in range(CONFIG['n_topics']):
        n_cells = np.sum(cluster_labels == k)
        percentage = n_cells / len(cluster_labels) * 100
        
        # 获取该聚类的关键基因
        cluster_genes = top_genes[k][:3]
        gene_names = [g[0] for g in cluster_genes]
        
        print(f"  聚类 {k}: {n_cells} 个细胞 ({percentage:.1f}%)")
        print(f"      关键基因: {', '.join(gene_names)}")
    
    # 6. 可视化
    print("\n5. 可视化结果...")
    visualizer = ClusteringVisualizer()
    
    # 空间聚类图
    visualizer.plot_spatial_clusters(
        positions, 
        cluster_labels,
        title=f"Visium HD LDA Clustering (K={CONFIG['n_topics']})"
    )
    
    # 主题分布热图
    visualizer.plot_topic_distributions(lda.theta)
    
    # 收敛曲线
    if len(lda.log_likelihoods) > 0:
        visualizer.plot_convergence(lda.log_likelihoods)
    
    # 关键基因图
    visualizer.plot_top_genes(top_genes, n_show=5)
    
    # 7. 保存结果
    print("\n6. 保存结果...")
    
    # 创建结果DataFrame
    results_df = pd.DataFrame({
        'cell_id': expr_matrix.index[:len(cluster_labels)],
        'x': positions[:, 0],
        'y': positions[:, 1],
        'cluster': cluster_labels
    })
    
    # 添加主题概率
    for k in range(CONFIG['n_topics']):
        results_df[f'topic_{k}_prob'] = lda.theta[:, k]
    
    results_file = 'lda_clustering_results.csv'
    results_df.to_csv(results_file, index=False)
    print(f"聚类结果已保存到: {results_file}")
    
    # 保存主题-基因分布
    topic_genes_df = pd.DataFrame(
        lda.beta.T,
        columns=[f'topic_{k}' for k in range(CONFIG['n_topics'])],
        index=processor.gene_names
    )
    topic_genes_file = 'topic_gene_distributions.csv'
    topic_genes_df.to_csv(topic_genes_file)
    print(f"主题-基因分布已保存到: {topic_genes_file}")
    
    # 保存关键基因
    top_genes_list = []
    for k in range(CONFIG['n_topics']):
        for gene_name, prob in top_genes[k]:
            top_genes_list.append({
                'topic': k,
                'gene': gene_name,
                'probability': prob
            })
    
    top_genes_df = pd.DataFrame(top_genes_list)
    top_genes_file = 'top_genes_by_topic.csv'
    top_genes_df.to_csv(top_genes_file, index=False)
    print(f"关键基因列表已保存到: {top_genes_file}")
    
    print("\n" + "=" * 60)
    print("分析完成!")
    print("=" * 60)

# ==================== 简化的测试版本（用于快速验证） ====================
def test_small_data():
    """测试小规模数据"""
    print("测试小规模数据...")
    
    # 只处理前1000个细胞和前100个基因，加快速度
    expr_file = 'Task1-Visium_HD-HVG_rank_expr.txt'
    position_file = 'Task1-Visium_HD-Position.txt'
    
    # 加载数据
    df = pd.read_csv(expr_file, sep='\t', index_col=0)
    
    # 取子集
    df_subset = df.iloc[:100, :1000]  # 前100个基因，前1000个细胞
    df_subset = df_subset.T  # 转置为细胞×基因
    
    processor = VisiumDataProcessor(n_bins=3)
    processor.gene_names = list(df_subset.columns)
    processor.vocab = {gene: idx for idx, gene in enumerate(processor.gene_names)}
    
    # 离散化
    discretized = processor.discretize_expression(df_subset, method='quantile')
    
    # 创建文档
    documents = processor.create_documents(discretized, min_expression=1)
    
    # 加载位置（只取对应的细胞）
    positions_df = pd.read_csv(position_file, sep='\t')
    positions = positions_df.iloc[:1000, 1:3].values
    
    # 训练模型
    lda = SpatialLDA(
        n_topics=4,
        alpha=0.1,
        eta=0.01,
        spatial_smooth=0.2,
        n_iter=100,
        burn_in=20
    )
    
    lda.fit(
        documents=documents,
        vocab_size=len(processor.gene_names),
        positions=positions
    )
    
    # 聚类
    cluster_labels = lda.cluster_cells(method='hard')
    
    # 可视化
    visualizer = ClusteringVisualizer()
    visualizer.plot_spatial_clusters(
        positions, 
        cluster_labels,
        title="Test: Visium HD LDA Clustering"
    )
    
    return lda, cluster_labels, processor.gene_names

if __name__ == "__main__":
    # 完整分析
    main()
    
    # 或者运行测试版本
    # lda, labels, gene_names = test_small_data()