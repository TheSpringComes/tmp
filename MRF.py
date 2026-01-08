import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import time

class MRFImageSegmentation:
    """
    基于MRF模型的病理图像分割
    """
    
    def __init__(self, K=3, beta=1.0, max_iter=50, tolerance=1e-4):
        """
        初始化MRF分割模型
        
        参数:
        K: 类别数
        beta: 耦合系数（平滑参数）
        max_iter: 最大迭代次数
        tolerance: 收敛阈值
        """
        self.K = K
        self.beta = beta
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.height = None
        self.width = None
        self.channels = None
        
    def read_image(self, image_path, resize_factor=1.0):
        """
        读取病理图像并进行预处理
        
        参数:
        image_path: 图像路径
        resize_factor: 缩放因子
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        # BGR转RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 调整大小
        if resize_factor != 1.0:
            new_height = int(image.shape[0] * resize_factor)
            new_width = int(image.shape[1] * resize_factor)
            image = cv2.resize(image, (new_width, new_height))
        
        # 归一化到[0, 1]
        self.image = image.astype(np.float32) / 255.0
        
        # 获取图像尺寸
        self.height, self.width, self.channels = self.image.shape
        
        print(f"图像尺寸: {self.height} x {self.width}, 通道数: {self.channels}")
        
        return self.image
    
    def initialize_labels(self, method='kmeans'):
        """
        初始化标签场Z
        
        参数:
        method: 初始化方法 ('kmeans', 'gmm', 'random')
        """
        # 将图像展平为像素数组
        pixels = self.image.reshape(-1, self.channels)
        
        if method == 'kmeans':
            # 使用K-means聚类初始化
            kmeans = KMeans(n_clusters=self.K, random_state=42, n_init=10)
            labels = kmeans.fit_predict(pixels)
            
            # 计算每个类别的初始均值和协方差
            self.means = []
            self.covariances = []
            
            for k in range(self.K):
                mask = (labels == k)
                if np.sum(mask) > 0:
                    cluster_pixels = pixels[mask]
                    self.means.append(np.mean(cluster_pixels, axis=0))
                    self.covariances.append(np.cov(cluster_pixels.T) + 1e-6 * np.eye(self.channels))
                else:
                    # 如果某个类别没有像素，使用随机均值和单位协方差
                    self.means.append(np.random.rand(self.channels))
                    self.covariances.append(np.eye(self.channels))
                    
        elif method == 'gmm':
            # 使用高斯混合模型初始化
            gmm = GaussianMixture(n_components=self.K, random_state=42)
            gmm.fit(pixels)
            labels = gmm.predict(pixels)
            self.means = gmm.means_
            self.covariances = gmm.covariances_
            
        else:  # random
            # 随机初始化
            labels = np.random.randint(0, self.K, size=pixels.shape[0])
            
            # 计算每个类别的初始均值和协方差
            self.means = []
            self.covariances = []
            
            for k in range(self.K):
                mask = (labels == k)
                if np.sum(mask) > 0:
                    cluster_pixels = pixels[mask]
                    self.means.append(np.mean(cluster_pixels, axis=0))
                    self.covariances.append(np.cov(cluster_pixels.T) + 1e-6 * np.eye(self.channels))
                else:
                    self.means.append(np.random.rand(self.channels))
                    self.covariances.append(np.eye(self.channels))
        
        # 将标签重塑为图像形状
        self.Z = labels.reshape(self.height, self.width)
        
        # 转换为整数类型
        self.Z = self.Z.astype(np.int32)
        
        print(f"标签初始化完成，类别数: {self.K}")
        
        return self.Z
    
    def compute_pairwise_potential(self, z1, z2):
        """
        计算成对势能 U(z_i, z_j)
        
        根据Potts模型:
        U(z_i, z_j) = -β if z_i = z_j
                      +β if z_i ≠ z_j
        """
        if z1 == z2:
            return -self.beta
        else:
            return self.beta
    
    def compute_gaussian_log_likelihood(self, x, k):
        """
        计算高斯分布的对数似然: log N(x | μ_k, Σ_k)
        """
        try:
            # 使用多元高斯分布计算对数似然
            mvn = multivariate_normal(mean=self.means[k], cov=self.covariances[k])
            return mvn.logpdf(x)
        except:
            # 如果协方差矩阵有问题，返回一个较小的值
            return -1e10
    
    def compute_energy_for_pixel(self, i, j, k):
        """
        计算像素(i,j)标记为类别k时的局部能量 E(Z_i=k)
        """
        # 1. 观测似然项: -log P(X_i | Z_i=k)
        x = self.image[i, j]
        likelihood_energy = -self.compute_gaussian_log_likelihood(x, k)
        
        # 2. 先验项: 与邻域像素的成对势能之和
        prior_energy = 0.0
        
        # 4邻域系统
        neighbors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
        
        for ni, nj in neighbors:
            if 0 <= ni < self.height and 0 <= nj < self.width:
                prior_energy += self.compute_pairwise_potential(k, self.Z[ni, nj])
        
        # 总能量 = 似然项 + 先验项
        total_energy = likelihood_energy + prior_energy
        
        return total_energy
    
    def icm_update(self):
        """
        执行一次ICM迭代更新
        返回: 标签变化的像素比例
        """
        # 创建新的标签场
        Z_new = self.Z.copy()
        
        # 记录变化的像素数
        changed_pixels = 0
        
        # 遍历所有像素
        for i in range(self.height):
            for j in range(self.width):
                # 计算当前像素选择每个类别的能量
                energies = []
                for k in range(self.K):
                    energy = self.compute_energy_for_pixel(i, j, k)
                    energies.append(energy)
                
                # 选择能量最小的类别
                new_label = np.argmin(energies)
                
                # 如果标签变化，记录
                if new_label != Z_new[i, j]:
                    changed_pixels += 1
                    Z_new[i, j] = new_label
        
        # 更新标签场
        self.Z = Z_new
        
        # 计算变化比例
        change_rate = changed_pixels / (self.height * self.width)
        
        return change_rate
    
    def update_gaussian_parameters(self):
        """
        更新高斯分布参数: 均值和协方差矩阵
        """
        # 将图像展平
        pixels = self.image.reshape(-1, self.channels)
        labels = self.Z.reshape(-1)
        
        # 更新每个类别的参数
        for k in range(self.K):
            # 获取属于类别k的像素
            mask = (labels == k)
            
            if np.sum(mask) > 1:  # 至少需要2个点来计算协方差
                cluster_pixels = pixels[mask]
                
                # 更新均值
                self.means[k] = np.mean(cluster_pixels, axis=0)
                
                # 更新协方差矩阵，添加正则项避免奇异
                cov_matrix = np.cov(cluster_pixels.T)
                
                # 如果协方差矩阵不是正定的，添加正则化
                if cov_matrix.shape == (self.channels, self.channels):
                    try:
                        # 尝试Cholesky分解检查正定性
                        np.linalg.cholesky(cov_matrix)
                        self.covariances[k] = cov_matrix
                    except np.linalg.LinAlgError:
                        # 添加正则化项
                        self.covariances[k] = cov_matrix + 1e-3 * np.eye(self.channels)
                else:
                    self.covariances[k] = np.eye(self.channels)
    
    def fit(self, image_path, init_method='kmeans', update_params=True):
        """
        训练MRF分割模型
        
        参数:
        image_path: 图像路径
        init_method: 初始化方法
        update_params: 是否更新高斯参数
        """
        # 1. 读取图像
        print("步骤1: 读取图像...")
        self.read_image(image_path)
        
        # 2. 初始化标签场
        print("步骤2: 初始化标签场...")
        self.initialize_labels(method=init_method)
        
        # 3. ICM迭代优化
        print("步骤3: ICM迭代优化...")
        
        history = {
            'change_rates': [],
            'energies': [],
            'iterations': []
        }
        
        start_time = time.time()
        
        for iteration in range(self.max_iter):
            print(f"  迭代 {iteration+1}/{self.max_iter}")
            
            # 执行ICM更新
            change_rate = self.icm_update()
            
            # 记录变化率
            history['change_rates'].append(change_rate)
            history['iterations'].append(iteration+1)
            
            # 如果需要，更新高斯参数
            if update_params and iteration % 2 == 0:  # 每2次迭代更新一次参数
                self.update_gaussian_parameters()
            
            # 检查收敛
            if change_rate < self.tolerance:
                print(f"  收敛于迭代 {iteration+1}, 变化率: {change_rate:.6f}")
                break
        
        end_time = time.time()
        print(f"ICM优化完成，耗时: {end_time-start_time:.2f}秒")
        
        return history
    
    def segment_image(self):
        """
        获取分割结果
        返回: 分割后的标签图像
        """
        return self.Z
    
    def visualize_results(self, save_path=None):
        """
        可视化原始图像和分割结果
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 原始图像
        axes[0].imshow(self.image)
        axes[0].set_title('原始病理图像')
        axes[0].axis('off')
        
        # 分割结果（伪彩色）
        cmap = plt.cm.get_cmap('viridis', self.K)
        seg_display = cmap(self.Z / (self.K-1) if self.K > 1 else self.Z)
        axes[1].imshow(seg_display)
        axes[1].set_title(f'MRF分割结果 (K={self.K}, β={self.beta})')
        axes[1].axis('off')
        
        # 叠加显示
        overlay = self.image.copy()
        # 为分割边界添加轮廓
        from scipy import ndimage
        
        # 计算每个类别的边界
        boundaries = np.zeros((self.height, self.width), dtype=bool)
        for k in range(self.K):
            mask = (self.Z == k)
            if np.any(mask):
                # 使用形态学梯度检测边界
                eroded = ndimage.binary_erosion(mask, structure=np.ones((3,3)))
                boundary = mask & ~eroded
                boundaries = boundaries | boundary
        
        # 在边界处标记为红色
        overlay[boundaries, :] = [1.0, 0.0, 0.0]  # 红色边界
        
        axes[2].imshow(overlay)
        axes[2].set_title('分割边界叠加')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"结果已保存到: {save_path}")
        
        plt.show()
    
    def evaluate_segmentation(self, ground_truth=None):
        """
        评估分割结果（如果有ground truth）
        
        参数:
        ground_truth: 真实分割标签
        
        返回: 评估指标
        """
        if ground_truth is None:
            print("未提供ground truth，无法进行定量评估")
            return None
        
        # 确保ground truth与分割结果尺寸一致
        if ground_truth.shape != self.Z.shape:
            print("ground truth尺寸不匹配，尝试调整...")
            ground_truth = cv2.resize(ground_truth, (self.width, self.height), 
                                      interpolation=cv2.INTER_NEAREST)
        
        # 计算准确率
        accuracy = np.mean(self.Z == ground_truth)
        
        # 计算互信息
        from sklearn.metrics import mutual_info_score
        mi = mutual_info_score(self.Z.flatten(), ground_truth.flatten())
        
        # 计算调整兰德指数
        from sklearn.metrics import adjusted_rand_score
        ari = adjusted_rand_score(self.Z.flatten(), ground_truth.flatten())
        
        metrics = {
            'accuracy': accuracy,
            'mutual_information': mi,
            'adjusted_rand_index': ari
        }
        
        print(f"分割评估结果:")
        print(f"  准确率: {accuracy:.4f}")
        print(f"  互信息: {mi:.4f}")
        print(f"  调整兰德指数: {ari:.4f}")
        
        return metrics


def main():
    """
    主函数：演示MRF病理图像分割的完整流程
    """
    # 创建MRF分割器实例
    mrf_seg = MRFImageSegmentation(
        K=3,          # 假设分割为3个区域（背景、正常、病变）
        beta=1.0,     # 耦合系数
        max_iter=30,  # 最大迭代次数
        tolerance=1e-5 # 收敛阈值
    )
    
    # 执行分割
    try:
        history = mrf_seg.fit(
            image_path='PGMProject2025-data/Task2-Visium_HD-Crop.png',
            init_method='kmeans',
            update_params=True
        )
    except Exception as e:
        print(f"图像读取失败: {e}")
        print("使用示例图像进行演示...")
        # 创建示例图像
        height, width = 200, 300
        example_image = np.zeros((height, width, 3))
        
        # 创建三个区域
        example_image[50:150, 50:150, :] = [0.7, 0.3, 0.3]  # 红色区域
        example_image[30:120, 180:280, :] = [0.3, 0.7, 0.3]  # 绿色区域
        example_image = example_image + 0.1 * np.random.randn(height, width, 3)  # 添加噪声
        
        # 保存示例图像
        example_image_path = 'example_pathology.png'
        cv2.imwrite(example_image_path, (example_image * 255).astype(np.uint8))
        
        # 使用示例图像
        mrf_seg.image = example_image
        mrf_seg.height, mrf_seg.width, mrf_seg.channels = example_image.shape
        
        # 初始化并运行
        mrf_seg.initialize_labels(method='kmeans')
        history = {'change_rates': [], 'iterations': []}
        
        for iteration in range(mrf_seg.max_iter):
            change_rate = mrf_seg.icm_update()
            history['change_rates'].append(change_rate)
            history['iterations'].append(iteration+1)
            
            if iteration % 2 == 0:
                mrf_seg.update_gaussian_parameters()
            
            if change_rate < mrf_seg.tolerance:
                break
    
    # 可视化结果
    mrf_seg.visualize_results(save_path='mrf_segmentation_result.png')
    
    # 显示迭代过程
    if history['change_rates']:
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['iterations'], history['change_rates'], 'b-o', linewidth=2)
        plt.xlabel('迭代次数')
        plt.ylabel('标签变化率')
        plt.title('ICM收敛过程')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        # 显示每个类别的均值
        colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow']
        for k in range(min(mrf_seg.K, len(colors))):
            plt.plot(mrf_seg.means[k][0], mrf_seg.means[k][1], 
                     'o', color=colors[k], markersize=10, label=f'类别{k}')
        plt.xlabel('R通道均值')
        plt.ylabel('G通道均值')
        plt.title('各类别颜色中心')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('convergence_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    # 参数敏感性分析
    print("\n参数敏感性分析:")
    
    # 测试不同的K值
    K_values = [2, 3, 4, 5]
    beta_fixed = 1.0
    
    plt.figure(figsize=(15, 10))
    
    for idx, K in enumerate(K_values):
        mrf_test = MRFImageSegmentation(K=K, beta=beta_fixed, max_iter=20)
        mrf_test.image = mrf_seg.image
        mrf_test.height, mrf_test.width, mrf_test.channels = mrf_seg.image.shape
        
        mrf_test.initialize_labels(method='kmeans')
        
        # 运行少量迭代
        for _ in range(10):
            mrf_test.icm_update()
            mrf_test.update_gaussian_parameters()
        
        plt.subplot(2, 2, idx+1)
        cmap = plt.cm.get_cmap('viridis', K)
        seg_display = cmap(mrf_test.Z / (K-1) if K > 1 else mrf_test.Z)
        plt.imshow(seg_display)
        plt.title(f'K={K}, β={beta_fixed}')
        plt.axis('off')
    
    plt.suptitle('不同类别数K对分割结果的影响', fontsize=16)
    plt.tight_layout()
    plt.savefig('K_sensitivity.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 测试不同的beta值
    beta_values = [0.1, 0.5, 1.0, 2.0]
    K_fixed = 3
    
    plt.figure(figsize=(15, 10))
    
    for idx, beta in enumerate(beta_values):
        mrf_test = MRFImageSegmentation(K=K_fixed, beta=beta, max_iter=20)
        mrf_test.image = mrf_seg.image
        mrf_test.height, mrf_test.width, mrf_test.channels = mrf_seg.image.shape
        
        mrf_test.initialize_labels(method='kmeans')
        
        # 运行少量迭代
        for _ in range(10):
            mrf_test.icm_update()
            mrf_test.update_gaussian_parameters()
        
        plt.subplot(2, 2, idx+1)
        cmap = plt.cm.get_cmap('viridis', K_fixed)
        seg_display = cmap(mrf_test.Z / (K_fixed-1) if K_fixed > 1 else mrf_test.Z)
        plt.imshow(seg_display)
        plt.title(f'K={K_fixed}, β={beta}')
        plt.axis('off')
    
    plt.suptitle('不同耦合系数β对分割结果的影响', fontsize=16)
    plt.tight_layout()
    plt.savefig('beta_sensitivity.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n分割完成！")
    print(f"最终分割标签形状: {mrf_seg.Z.shape}")
    print(f"标签分布: {np.bincount(mrf_seg.Z.flatten())}")

if __name__ == "__main__":
    main()