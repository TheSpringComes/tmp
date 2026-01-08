import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.special import gammaln, logsumexp
import os
import warnings
warnings.filterwarnings('ignore')

# 创建保存图片的目录
if not os.path.exists('output_images_bonus'):
    os.makedirs('output_images_bonus')

class SpatialImageLDA:
    """融合空间和图像信息的LDA模型"""
    
    def __init__(self, n_topics=8, alpha=0.1, eta=0.01, 
                 n_iter=100, burn_in=50, verbose=True,
                 lambda_s=1.0, lambda_i=0.5, n_neighbors=10):
        self.K = n_topics
        self.alpha = alpha
        self.eta = eta
        self.n_iter = n_iter
        self.burn_in = burn_in
        self.verbose = verbose
        self.lambda_s = lambda_s  # 空间一致性权重
        self.lambda_i = lambda_i  # 图像特征权重
        self.n_neighbors = n_neighbors  # 邻居数量
        
        # 图像特征相关
        self.image_features = None
        self.mu_k = None  # 主题k的图像特征原型
        
    def _initialize_structures(self, documents, positions=None, image_features=None):
        D = len(documents)
        
        # 存储文档的最大长度
        self.doc_lengths = np.zeros(D, dtype=np.int32)
        for d in range(D):
            self.doc_lengths[d] = sum(count for _, count in documents[d])
        
        # 初始化计数矩阵
        self.n_dk = np.zeros((D, self.K), dtype=np.int32)  # 文档-主题计数
        self.n_kv = np.zeros((self.K, self.V), dtype=np.int32)  # 主题-词计数
        self.n_k = np.zeros(self.K, dtype=np.int32)  # 主题总计数
        
        # 存储文档的词汇和计数（用于快速访问）
        self.doc_words = []
        self.doc_counts = []
        self.doc_start_pos = np.zeros(D, dtype=np.int32)
        
        # 构建文档表示
        current_pos = 0
        for d in range(D):
            words = []
            counts = []
            for word, count in documents[d]:
                words.append(word)
                counts.append(count)
            
            self.doc_words.append(np.array(words, dtype=np.int32))
            self.doc_counts.append(np.array(counts, dtype=np.int32))
            self.doc_start_pos[d] = current_pos
            current_pos += self.doc_lengths[d]
        
        # 初始化主题分配
        total_tokens = current_pos
        self.z_flat = np.random.randint(0, self.K, size=total_tokens)
        
        # 重建z的文档结构（用于调试）
        self.z_by_doc = []
        pos = 0
        for d in range(D):
            doc_len = self.doc_lengths[d]
            self.z_by_doc.append(self.z_flat[pos:pos+doc_len])
            pos += doc_len
        
        # 初始化邻居信息
        if positions is not None:
            self.positions = positions
            self._build_neighbors()
        
        # 初始化图像特征
        if image_features is not None:
            self.image_features = image_features
        else:
            # 如果没有提供图像特征，使用随机特征（模拟）
            self.image_features = np.random.randn(D, 10)  # 10维随机特征
        
        # 初始化主题原型
        self.mu_k = np.random.randn(self.K, self.image_features.shape[1])
        
        # 计算初始计数
        self._update_counts_from_z()
        
        # 计算初始邻居主题计数
        if positions is not None:
            self._update_neighbor_counts()
    
    def _build_neighbors(self):
        """构建邻居关系"""
        D = len(self.positions)
        tree = KDTree(self.positions)
        
        self.neighbors = []
        for d in range(D):
            # 找到k+1个最近邻（包括自己）
            distances, indices = tree.query(self.positions[d], k=self.n_neighbors+1)
            # 排除自己
            neighbor_indices = indices[1:]
            self.neighbors.append(neighbor_indices)
    
    def _update_counts_from_z(self):
        """从z更新计数矩阵"""
        self.n_dk.fill(0)
        self.n_kv.fill(0)
        self.n_k.fill(0)
        
        pos = 0
        for d in range(self.D):  # 使用 self.D 而不是 D
            doc_words = self.doc_words[d]
            doc_counts = self.doc_counts[d]
            doc_len = self.doc_lengths[d]
            
            for word_idx, count in zip(doc_words, doc_counts):
                for _ in range(count):
                    k = self.z_flat[pos]
                    self.n_dk[d, k] += 1
                    self.n_kv[k, word_idx] += 1
                    self.n_k[k] += 1
                    pos += 1
    
    def _update_neighbor_counts(self):
        """更新邻居主题计数"""
        D = len(self.doc_words)
        self.n_nbr_dk = np.zeros((D, self.K), dtype=np.int32)
        
        for d in range(D):
            neighbors = self.neighbors[d]
            for nbr in neighbors:
                self.n_nbr_dk[d] += self.n_dk[nbr]
    
    def _update_topic_prototypes(self):
        """更新主题的图像特征原型"""
        D = len(self.doc_words)  # 添加这行定义 D
        feature_dim = self.image_features.shape[1]
        
        # 重置原型
        self.mu_k.fill(0)
        topic_counts = np.zeros(self.K, dtype=np.int32)
        
        # 收集属于每个主题的所有细胞特征
        topic_features = [[] for _ in range(self.K)]
        
        pos = 0
        for d in range(D):
            doc_len = self.doc_lengths[d]
            doc_topics = self.z_flat[pos:pos+doc_len]
            
            # 对于每个文档，使用其主要主题
            if len(doc_topics) > 0:
                # 可以选择使用文档中最常见的主题
                unique, counts = np.unique(doc_topics, return_counts=True)
                main_topic = unique[np.argmax(counts)]
                
                topic_features[main_topic].append(self.image_features[d])
                topic_counts[main_topic] += 1
                
            pos += doc_len
        
        # 计算原型
        for k in range(self.K):
            if len(topic_features[k]) > 0:
                features_array = np.array(topic_features[k])
                self.mu_k[k] = np.mean(features_array, axis=0)
            else:
                # 如果没有细胞属于该主题，保持原值
                pass
    
    def fit(self, documents, vocab_size, positions=None, image_features=None):
        """训练融合空间和图像信息的LDA模型"""
        self.D = len(documents)
        self.V = vocab_size
        
        if self.verbose:
            print(f"开始训练: D={self.D}, V={self.V}, K={self.K}")
            print(f"λ_s={self.lambda_s}, λ_i={self.lambda_i}, 邻居数={self.n_neighbors}")
        
        # 初始化数据结构
        self._initialize_structures(documents, positions, image_features)
        
        # 预计算常数
        alpha_sum = self.K * self.alpha
        eta_sum = self.V * self.eta
        
        # 吉布斯采样
        print("开始吉布斯采样...")
        self.log_likelihoods = []
        
        for iteration in range(self.n_iter):
            if self.verbose and iteration % 10 == 0:
                print(f"迭代 {iteration}/{self.n_iter}")
            
            # 每10次迭代更新一次主题原型
            if iteration % 10 == 0 and self.lambda_i > 0:
                self._update_topic_prototypes()
            
            # 遍历所有token
            pos = 0
            for d in range(self.D):
                doc_words = self.doc_words[d]
                doc_counts = self.doc_counts[d]
                doc_len = self.doc_lengths[d]
                
                for word_idx, count in zip(doc_words, doc_counts):
                    for _ in range(count):
                        # 当前主题
                        old_k = self.z_flat[pos]
                        
                        # 从计数中移除当前token
                        self.n_dk[d, old_k] -= 1
                        self.n_kv[old_k, word_idx] -= 1
                        self.n_k[old_k] -= 1
                        
                        # 更新邻居计数（当前细胞的主题变化影响其邻居）
                        if positions is not None and self.lambda_s > 0:
                            neighbors = self.neighbors[d]
                            for nbr in neighbors:
                                self.n_nbr_dk[nbr, old_k] -= 1
                        
                        # 计算条件概率
                        log_probs = np.zeros(self.K)
                        
                        # LDA核心项
                        for k in range(self.K):
                            # 第一项: (n_{d,k} + α)
                            term1 = np.log(self.n_dk[d, k] + self.alpha)
                            
                            # 第二项: (n_{k,w} + η) / (n_k + Vη)
                            term2 = np.log(self.n_kv[k, word_idx] + self.eta)
                            term2 -= np.log(self.n_k[k] + eta_sum)
                            
                            log_probs[k] = term1 + term2
                            
                            # 空间一致性项: λ_s * n_nbr_{d,k}
                            if positions is not None and self.lambda_s > 0:
                                log_probs[k] += self.lambda_s * self.n_nbr_dk[d, k]
                            
                            # 图像相似性项: -λ_i * ||f_d - μ_k||^2
                            if self.lambda_i > 0 and self.image_features is not None:
                                dist = np.sum((self.image_features[d] - self.mu_k[k]) ** 2)
                                log_probs[k] -= self.lambda_i * dist
                        
                        # 归一化概率
                        log_probs -= logsumexp(log_probs)
                        probs = np.exp(log_probs)
                        
                        # 采样新主题
                        new_k = np.random.choice(self.K, p=probs)
                        
                        # 更新主题分配
                        self.z_flat[pos] = new_k
                        
                        # 更新计数
                        self.n_dk[d, new_k] += 1
                        self.n_kv[new_k, word_idx] += 1
                        self.n_k[new_k] += 1
                        
                        # 更新邻居计数
                        if positions is not None and self.lambda_s > 0:
                            neighbors = self.neighbors[d]
                            for nbr in neighbors:
                                self.n_nbr_dk[nbr, new_k] += 1
                        
                        pos += 1
            
            # 计算对数似然
            if iteration >= self.burn_in and iteration % 5 == 0:
                ll = self._calculate_log_likelihood()
                self.log_likelihoods.append(ll)
                if self.verbose:
                    print(f"  迭代 {iteration}: LL = {ll:.2f}")
        
        # 估计参数
        self.theta, self.beta = self._estimate_parameters()
        
        if self.verbose:
            print("训练完成")
        
        return self
    
    def _calculate_log_likelihood(self):
        """计算对数似然"""
        log_lik = 0
        alpha_sum = self.K * self.alpha
        eta_sum = self.V * self.eta
        
        # 文档-主题部分
        for d in range(self.D):
            N_d = self.doc_lengths[d]
            for k in range(self.K):
                log_lik += gammaln(self.n_dk[d, k] + self.alpha)
            log_lik -= gammaln(N_d + alpha_sum)
        
        # 主题-词部分
        for k in range(self.K):
            for v in range(self.V):
                log_lik += gammaln(self.n_kv[k, v] + self.eta)
            log_lik -= gammaln(self.n_k[k] + eta_sum)
        
        return log_lik
    
    def _estimate_parameters(self):
        """估计参数"""
        D = len(self.doc_words)
        theta = np.zeros((D, self.K))
        alpha_sum = self.K * self.alpha
        
        for d in range(D):
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

class EnhancedDataProcessor:
    """增强的数据处理器，包括图像特征提取"""
    
    def __init__(self, top_genes=500, n_bins=3, min_expression=1):
        self.top_genes = top_genes
        self.n_bins = n_bins
        self.min_expression = min_expression
        self.gene_names = None
    
    def process(self, expr_file, position_file, n_cells=None, 
                image_feature_dim=10, extract_image_features=False):
        """处理数据，包括图像特征提取"""
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
        
        # 提取图像特征（模拟）
        print("6. 提取图像特征...")
        if extract_image_features:
            # 这里应该是从实际图像中提取特征
            # 为了演示，我们使用随机特征
            image_features = np.random.randn(len(documents), image_feature_dim)
        else:
            # 使用基于基因表达的特征（模拟图像特征）
            # 使用PCA降维到image_feature_dim维
            from sklearn.decomposition import PCA
            pca = PCA(n_components=image_feature_dim)
            image_features = pca.fit_transform(df.values)
        
        print(f"  图像特征维度: {image_features.shape[1]}")
        
        return documents, positions, image_features, self.gene_names

def run_bonus_analysis():
    """运行Bonus分析"""
    print("=" * 60)
    print("融合空间和图像信息的LDA聚类分析 (Bonus)")
    print("=" * 60)
    
    # 参数设置
    CONFIG = {
        'expr_file': 'PGMProject2025-data/Task1-Visium_HD-HVG_rank_expr.txt',
        'position_file': 'PGMProject2025-data/Task1-Visium_HD-Position.txt',
        'n_topics': 4,
        'top_genes': 200,
        'n_cells': 37817,  # 为了速度，使用部分细胞
        'n_bins': 5,
        'n_iter': 100,
        'burn_in': 50,
        'lambda_s': 0.5,  # 空间权重
        'lambda_i': 0.3,  # 图像权重
        'n_neighbors': 8,  # 邻居数量
        'image_feature_dim': 10,  # 图像特征维度
    }
    
    # 1. 处理数据
    print("\n阶段1: 数据预处理...")
    processor = EnhancedDataProcessor(
        top_genes=CONFIG['top_genes'],
        n_bins=CONFIG['n_bins'],
        min_expression=1
    )
    
    documents, positions, image_features, gene_names = processor.process(
        CONFIG['expr_file'],
        CONFIG['position_file'],
        n_cells=CONFIG['n_cells'],
        image_feature_dim=CONFIG['image_feature_dim'],
        extract_image_features=False
    )
    
    # 2. 训练基础LDA（用于对比）
    print("\n阶段2: 训练基础LDA模型用于对比...")
    
    class BasicLDA:
        def __init__(self, n_topics=8, alpha=0.1, eta=0.01, n_iter=100):
            self.K = n_topics
            self.alpha = alpha
            self.eta = eta
            self.n_iter = n_iter
        
        def fit(self, documents, vocab_size):
            self.D = len(documents)
            self.V = vocab_size
            
            # 初始化计数
            self.n_dk = np.zeros((self.D, self.K))
            self.n_kv = np.zeros((self.K, self.V))
            self.n_k = np.zeros(self.K)
            
            # 随机初始化
            self.z = []
            for d in range(self.D):
                doc_len = sum(count for _, count in documents[d])
                topics = np.random.randint(0, self.K, size=doc_len)
                self.z.append(topics)
                
                # 更新计数
                pos = 0
                for word_idx, count in documents[d]:
                    for _ in range(count):
                        k = topics[pos]
                        self.n_dk[d, k] += 1
                        self.n_kv[k, word_idx] += 1
                        self.n_k[k] += 1
                        pos += 1
            
            # 吉布斯采样
            for iteration in range(self.n_iter):
                for d in range(self.D):
                    pos = 0
                    for word_idx, count in documents[d]:
                        for _ in range(count):
                            old_k = self.z[d][pos]
                            
                            # 移除
                            self.n_dk[d, old_k] -= 1
                            self.n_kv[old_k, word_idx] -= 1
                            self.n_k[old_k] -= 1
                            
                            # 计算概率
                            probs = np.zeros(self.K)
                            for k in range(self.K):
                                term1 = self.n_dk[d, k] + self.alpha
                                term2 = (self.n_kv[k, word_idx] + self.eta) / (self.n_k[k] + self.V * self.eta)
                                probs[k] = term1 * term2
                            
                            probs /= probs.sum()
                            
                            # 采样
                            new_k = np.random.choice(self.K, p=probs)
                            self.z[d][pos] = new_k
                            
                            # 更新
                            self.n_dk[d, new_k] += 1
                            self.n_kv[new_k, word_idx] += 1
                            self.n_k[new_k] += 1
                            
                            pos += 1
            
            # 估计参数
            self.theta = np.zeros((self.D, self.K))
            for d in range(self.D):
                N_d = sum(count for _, count in documents[d])
                for k in range(self.K):
                    self.theta[d, k] = (self.n_dk[d, k] + self.alpha) / (N_d + self.K * self.alpha)
            
            return self
        
        def cluster_cells(self):
            return np.argmax(self.theta, axis=1)
    
    basic_lda = BasicLDA(n_topics=CONFIG['n_topics'], alpha=0.05, eta=0.01, n_iter=50)
    basic_lda.fit(documents, len(gene_names))
    basic_labels = basic_lda.cluster_cells()
    
    # 3. 训练增强LDA
    print("\n阶段3: 训练融合空间和图像信息的LDA模型...")
    enhanced_lda = SpatialImageLDA(
        n_topics=CONFIG['n_topics'],
        alpha=0.05,
        eta=0.01,
        n_iter=CONFIG['n_iter'],
        burn_in=CONFIG['burn_in'],
        verbose=True,
        lambda_s=CONFIG['lambda_s'],
        lambda_i=CONFIG['lambda_i'],
        n_neighbors=CONFIG['n_neighbors']
    )
    
    enhanced_lda.fit(documents, len(gene_names), positions, image_features)
    enhanced_labels = enhanced_lda.cluster_cells()
    
    # 4. 可视化对比
    print("\n阶段4: 可视化对比...")
    
    _, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # 图1: 基础LDA结果
    ax1 = axes[0]
    unique_labels = np.unique(basic_labels)
    colors = plt.cm.hsv(np.linspace(0, 1, len(unique_labels), endpoint=False))
    
    for label, color in zip(unique_labels, colors):
        mask = basic_labels == label
        ax1.scatter(positions[mask, 0], positions[mask, 1],
                   c=[color], label=f'Cluster {label}',
                   s=8, alpha=0.7, edgecolors='none')
    
    ax1.set_xlabel('X Coordinate', fontsize=12)
    ax1.set_ylabel('Y Coordinate', fontsize=12)
    ax1.set_title(f'Basic LDA (K={CONFIG["n_topics"]})', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')
    ax1.invert_yaxis()
    
    # 图2: 增强LDA结果
    ax2 = axes[1]
    unique_labels = np.unique(enhanced_labels)
    colors = plt.cm.hsv(np.linspace(0, 1, len(unique_labels), endpoint=False))
    
    for label, color in zip(unique_labels, colors):
        mask = enhanced_labels == label
        ax2.scatter(positions[mask, 0], positions[mask, 1],
                   c=[color], label=f'Cluster {label}',
                   s=8, alpha=0.7, edgecolors='none')
    
    ax2.set_xlabel('X Coordinate', fontsize=12)
    ax2.set_ylabel('Y Coordinate', fontsize=12)
    ax2.set_title(fr'Enhanced LDA ($λ_s$={CONFIG["lambda_s"]}, $λ_i$={CONFIG["lambda_i"]})', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='box')
    ax2.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('output_images_bonus/01_comparison.png', dpi=300, bbox_inches='tight')
    print("  已保存: output_images_bonus/01_comparison.png")
    plt.show()
    
    
    # ---------------------------
    # 2. 保存单独的基础LDA图
    # ---------------------------
    fig_basic, ax_basic = plt.subplots(figsize=(6, 6))
    unique_labels_basic = np.unique(basic_labels)
    colors_basic = plt.cm.hsv(np.linspace(0, 1, len(unique_labels_basic), endpoint=False))
    for label, color in zip(unique_labels_basic, colors_basic):
        mask = basic_labels == label
        ax_basic.scatter(positions[mask, 0], positions[mask, 1],
                        c=[color], label=f'Cluster {label}',
                        s=8, alpha=0.7, edgecolors='none')
    ax_basic.set_xlabel('X Coordinate', fontsize=12)
    ax_basic.set_ylabel('Y Coordinate', fontsize=12)
    ax_basic.set_title(f'Basic LDA (K={CONFIG["n_topics"]})', fontsize=14)
    ax_basic.grid(True, alpha=0.3)
    ax_basic.set_aspect('equal', adjustable='box')
    ax_basic.invert_yaxis()
    fig_basic.tight_layout()
    fig_basic.savefig('output_images_bonus/01_basic.png', dpi=300, bbox_inches='tight')
    plt.close(fig_basic)
    print("  已保存: output_images_bonus/01_basic.png")


    # ---------------------------
    # 3. 保存单独的增强LDA图
    # ---------------------------
    fig_enhanced, ax_enhanced = plt.subplots(figsize=(6, 6))
    unique_labels_enhanced = np.unique(enhanced_labels)
    colors_enhanced = plt.cm.hsv(np.linspace(0, 1, len(unique_labels_enhanced), endpoint=False))
    for label, color in zip(unique_labels_enhanced, colors_enhanced):
        mask = enhanced_labels == label
        ax_enhanced.scatter(positions[mask, 0], positions[mask, 1],
                            c=[color], label=f'Cluster {label}',
                            s=8, alpha=0.7, edgecolors='none')
    ax_enhanced.set_xlabel('X Coordinate', fontsize=12)
    ax_enhanced.set_ylabel('Y Coordinate', fontsize=12)
    ax_enhanced.set_title(fr'Enhanced LDA ($\lambda_s$={CONFIG["lambda_s"]}, $\lambda_i$={CONFIG["lambda_i"]})', fontsize=14)
    ax_enhanced.grid(True, alpha=0.3)
    ax_enhanced.set_aspect('equal', adjustable='box')
    ax_enhanced.invert_yaxis()
    fig_enhanced.tight_layout()
    fig_enhanced.savefig('output_images_bonus/01_enhanced.png', dpi=300, bbox_inches='tight')
    plt.close(fig_enhanced)
    print("  已保存: output_images_bonus/01_enhanced.png")
    
    
    # 5. 聚类质量评估
    print("\n阶段5: 聚类质量评估...")
    
    def calculate_spatial_coherence(labels, positions):
        """计算空间一致性：邻居具有相同标签的比例"""
        from scipy.spatial import KDTree
        
        tree = KDTree(positions)
        total_neighbors = 0
        same_label_neighbors = 0
        
        for i in range(len(labels)):
            # 找到5个最近邻（排除自己）
            distances, indices = tree.query(positions[i], k=6)
            neighbors = indices[1:]  # 排除第一个（自己）
            
            for nbr in neighbors:
                total_neighbors += 1
                if labels[i] == labels[nbr]:
                    same_label_neighbors += 1
        
        return same_label_neighbors / total_neighbors if total_neighbors > 0 else 0
    
    # 计算空间一致性
    basic_coherence = calculate_spatial_coherence(basic_labels, positions)
    enhanced_coherence = calculate_spatial_coherence(enhanced_labels, positions)
    
    print(f"空间一致性评估:")
    print(f"  基础LDA: {basic_coherence:.4f}")
    print(f"  增强LDA: {enhanced_coherence:.4f}")
    print(f"  提升: {(enhanced_coherence - basic_coherence)/basic_coherence*100:.1f}%")
    
    # 6. 保存结果
    print("\n阶段6: 保存结果...")
    
    # 保存聚类结果
    results_df = pd.DataFrame({
        'cell_id': range(len(enhanced_labels)),
        'x': positions[:, 0],
        'y': positions[:, 1],
        'basic_lda_cluster': basic_labels,
        'enhanced_lda_cluster': enhanced_labels
    })
    
    # 添加主题概率
    for k in range(CONFIG['n_topics']):
        results_df[f'topic_{k}_prob'] = enhanced_lda.theta[:, k]
    
    results_df.to_csv('output_images_bonus/02_clustering_results.csv', index=False)
    print(f"  已保存: output_images_bonus/02_clustering_results.csv")
    
    # 保存参数设置
    config_df = pd.DataFrame([CONFIG])
    config_df.to_csv('output_images_bonus/03_experiment_config.csv', index=False)
    print(f"  已保存: output_images_bonus/03_experiment_config.csv")
    
    # 保存评估结果
    eval_df = pd.DataFrame({
        'model': ['Basic LDA', 'Enhanced LDA'],
        'spatial_coherence': [basic_coherence, enhanced_coherence],
        'lambda_s': [0, CONFIG['lambda_s']],
        'lambda_i': [0, CONFIG['lambda_i']]
    })
    eval_df.to_csv('output_images_bonus/04_evaluation_results.csv', index=False)
    print(f"  已保存: output_images_bonus/04_evaluation_results.csv")
    
    print("\n" + "=" * 60)
    print("Bonus分析完成！所有结果已保存在 output_images_bonus/ 目录下")
    print("=" * 60)
    
    return enhanced_lda, basic_labels, enhanced_labels, gene_names


if __name__ == "__main__":
    enhanced_lda, basic_labels, enhanced_labels, genes = run_bonus_analysis()