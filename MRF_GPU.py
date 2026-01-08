import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import os

class MRFGPUSegmentation:
    """
    基于PyTorch GPU的MRF病理图像分割
    """
    
    def __init__(self, K=4, beta=1.5, max_iter=30, tolerance=1e-5):
        """
        初始化MRF GPU分割模型
        
        参数:
        K: 类别数
        beta: 耦合系数
        max_iter: 最大迭代次数
        tolerance: 收敛阈值
        """
        self.K = K
        self.beta = beta
        self.max_iter = max_iter
        self.tolerance = tolerance
        
        # 设置计算设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # 模型参数
        self.means = None
        self.covariances = None
        self.inv_covs = None
        self.log_dets = None
        
        # 数据张量
        self.X_tensor = None
        self.Z_tensor = None
        self.height = None
        self.width = None
        self.channels = None
        
    def load_image(self, image_path):
        """
        加载图像到GPU
        """
        # 读取图像
        if not os.path.exists(image_path):
            print(f"图像不存在: {image_path}，创建示例图像...")
            return self._create_sample_image()
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图像: {image_path}，创建示例图像...")
            return self._create_sample_image()
        
        # BGR转RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 转换为float32并归一化
        image = image.astype(np.float32) / 255.0
        
        # 转换为PyTorch张量 [C, H, W] 格式
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        
        # 移动到设备
        self.X_tensor = image_tensor.to(self.device)
        
        # 获取尺寸
        self.channels, self.height, self.width = image_tensor.shape
        
        print(f"图像尺寸: {self.height} × {self.width} × {self.channels}")
        print(f"总像素数: {self.height * self.width:,}")
        
        return self.X_tensor.cpu().permute(1, 2, 0).numpy()
    
    def _create_sample_image(self):
        """创建示例病理图像"""
        height, width = 512, 512
        
        # 创建示例病理图像
        image = np.zeros((height, width, 3), dtype=np.float32)
        
        # 添加不同组织区域
        # 细胞核区域（深色）
        image[100:200, 100:200, :] = [0.3, 0.1, 0.5]
        # 细胞质区域（中等色）
        image[150:300, 200:350, :] = [0.8, 0.6, 0.7]
        # 背景区域（浅色）
        image[250:400, 100:300, :] = [0.9, 0.9, 0.9]
        # 其他组织（其他颜色）
        image[350:500, 300:500, :] = [0.7, 0.8, 0.6]
        
        # 添加高斯噪声
        noise = np.random.normal(0, 0.05, (height, width, 3))
        image = np.clip(image + noise, 0, 1)
        
        # 转换为PyTorch张量
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        
        # 移动到设备
        self.X_tensor = image_tensor.to(self.device)
        
        # 获取尺寸
        self.channels, self.height, self.width = image_tensor.shape
        
        print(f"创建示例图像尺寸: {self.height} × {self.width} × {self.channels}")
        
        return image
    
    def initialize(self):
        """
        使用GPU K-means初始化标签场
        """
        # 将图像展平为 [N, C]
        X_flat = self.X_tensor.permute(1, 2, 0).reshape(-1, self.channels)
        N = X_flat.shape[0]
        
        print("使用GPU K-means初始化...")
        
        # 随机选择初始聚类中心
        indices = torch.randperm(N)[:self.K].to(self.device)
        centers = X_flat[indices].clone()
        
        # 运行K-means迭代
        for i in range(10):
            # 计算距离 [N, K]
            dists = torch.cdist(X_flat, centers, p=2)
            
            # 分配标签
            labels = torch.argmin(dists, dim=1)
            
            # 更新聚类中心
            new_centers = torch.zeros_like(centers)
            counts = torch.zeros(self.K, device=self.device)
            
            for k in range(self.K):
                mask = labels == k
                count = mask.sum().item()
                if count > 0:
                    new_centers[k] = X_flat[mask].mean(dim=0)
                    counts[k] = count
            
            # 如果有空的聚类中心，用随机点重新初始化
            empty_clusters = (counts == 0)
            if empty_clusters.any():
                for k in range(self.K):
                    if empty_clusters[k]:
                        new_centers[k] = X_flat[torch.randint(0, N, (1,))].squeeze()
            
            # 检查收敛
            if torch.allclose(centers, new_centers, rtol=1e-4):
                break
            
            centers = new_centers
        
        # 最终标签分配
        dists = torch.cdist(X_flat, centers, p=2)
        labels = torch.argmin(dists, dim=1)
        
        # 重塑为图像形状
        self.Z_tensor = labels.view(self.height, self.width)
        
        # 计算初始高斯参数
        self._compute_gaussian_parameters()
        
        print(f"初始化完成，类别数: {self.K}")
        
        return self.Z_tensor.cpu().numpy()
    
    def _compute_gaussian_parameters(self):
        """
        计算高斯分布参数（均值和协方差）
        """
        # 将图像展平
        X_flat = self.X_tensor.permute(1, 2, 0).reshape(-1, self.channels)
        Z_flat = self.Z_tensor.view(-1)
        
        # 初始化参数张量
        self.means = torch.zeros((self.K, self.channels), device=self.device)
        self.covariances = torch.zeros((self.K, self.channels, self.channels), device=self.device)
        
        # 计算每个类别的参数
        for k in range(self.K):
            mask = Z_flat == k
            count = mask.sum().item()
            
            if count > 0:
                # 提取属于该类别的像素
                X_k = X_flat[mask]
                
                # 计算均值
                mean_k = X_k.mean(dim=0)
                self.means[k] = mean_k
                
                # 计算协方差
                if count > 1:
                    diff = X_k - mean_k.unsqueeze(0)
                    # 使用矩阵乘法计算协方差
                    cov_k = torch.mm(diff.t(), diff) / (count - 1)
                    # 添加正则化确保正定性
                    cov_k = cov_k + 1e-3 * torch.eye(self.channels, device=self.device)
                else:
                    cov_k = torch.eye(self.channels, device=self.device)
                
                self.covariances[k] = cov_k
        
        # 预计算逆矩阵和行列式
        self._precompute_inverses()
    
    def _precompute_inverses(self):
        """
        预计算协方差矩阵的逆矩阵和行列式
        """
        self.inv_covs = torch.zeros((self.K, self.channels, self.channels), device=self.device)
        self.log_dets = torch.zeros(self.K, device=self.device)
        
        for k in range(self.K):
            cov = self.covariances[k]
            
            # 使用Cholesky分解求逆和行列式
            try:
                L = torch.linalg.cholesky(cov)
                # 计算逆矩阵
                inv_cov = torch.cholesky_inverse(L)
                # 计算行列式的对数
                log_det = 2 * torch.sum(torch.log(torch.diag(L)))
            except:
                # 如果Cholesky失败，使用伪逆
                inv_cov = torch.linalg.pinv(cov)
                # 计算行列式（添加正则化）
                reg_cov = cov + 1e-3 * torch.eye(self.channels, device=self.device)
                log_det = torch.logdet(reg_cov)
            
            self.inv_covs[k] = inv_cov
            self.log_dets[k] = log_det
    
    def compute_log_likelihood(self):
        """
        计算所有像素对所有类别的对数似然
        返回形状为 [K, H, W] 的张量
        """
        H, W = self.height, self.width
        K = self.K
        
        # 重塑图像为 [1, C, N]，其中 N = H*W
        X_flat = self.X_tensor.view(1, self.channels, -1)  # [1, C, N]
        N = X_flat.shape[2]
        
        # 扩展均值张量为 [K, C, 1]
        means_exp = self.means.unsqueeze(-1)  # [K, C, 1]
        
        # 计算差异 [K, C, N]
        # 使用广播：X_flat [1, C, N] - means_exp [K, C, 1] = [K, C, N]
        diff = X_flat - means_exp
        
        # 计算二次型：(x-μ)^T Σ^{-1} (x-μ)
        # 方法：先计算 Σ^{-1} (x-μ)
        # 重塑inv_covs为 [K, C, C]
        inv_covs_exp = self.inv_covs  # [K, C, C]
        
        # 使用批处理矩阵乘法计算 diff_inv = Σ^{-1} (x-μ)
        # diff形状: [K, C, N]，inv_covs_exp形状: [K, C, C]
        # 我们需要对每个K单独计算，所以使用torch.matmul
        diff_inv = torch.matmul(inv_covs_exp, diff.expand(K, -1, -1))
        
        # 计算二次型：sum((x-μ) * diff_inv)
        quad_form = torch.sum(diff.expand(K, -1, -1) * diff_inv, dim=1)  # [K, N]
        
        # 计算常数项
        const_term = self.channels * np.log(2 * np.pi) + self.log_dets.unsqueeze(-1)  # [K, 1]
        
        # 计算对数似然
        log_likelihood = -0.5 * (const_term + quad_form)  # [K, N]
        
        # 重塑为图像形状
        log_likelihood = log_likelihood.view(K, H, W)
        
        return log_likelihood
    
    def compute_prior_energy_vectorized(self):
        """
        向量化计算先验能量
        返回形状为 [K, H, W] 的张量
        """
        H, W = self.height, self.width
        K = self.K
        
        # 获取当前标签场
        Z = self.Z_tensor
        
        # 创建扩展的标签场用于向量化计算
        Z_padded = F.pad(Z.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='constant', value=-1)  # [1, 1, H+2, W+2]
        Z_padded = Z_padded.squeeze(0).squeeze(0)  # [H+2, W+2]
        
        # 计算每个位置的邻居标签
        # 上邻居
        Z_up = Z_padded[:-2, 1:-1]  # [H, W]
        # 下邻居
        Z_down = Z_padded[2:, 1:-1]  # [H, W]
        # 左邻居
        Z_left = Z_padded[1:-1, :-2]  # [H, W]
        # 右邻居
        Z_right = Z_padded[1:-1, 2:]  # [H, W]
        
        # 初始化先验能量张量
        prior_energy = torch.zeros(K, H, W, device=self.device)
        
        # 为每个类别计算先验能量
        for k in range(K):
            # 创建当前类别的掩码
            mask_k = (Z == k).float()  # [H, W]
            
            # 计算与当前类别相同的邻居数
            same_up = (Z_up == k).float()
            same_down = (Z_down == k).float()
            same_left = (Z_left == k).float()
            same_right = (Z_right == k).float()
            
            # 边界处理：无效邻居不计入
            valid_up = (Z_up != -1).float()
            valid_down = (Z_down != -1).float()
            valid_left = (Z_left != -1).float()
            valid_right = (Z_right != -1).float()
            
            # 计算相同邻居数
            same_count = same_up * valid_up + same_down * valid_down + same_left * valid_left + same_right * valid_right
            
            # 计算总有效邻居数
            total_neighbors = valid_up + valid_down + valid_left + valid_right
            
            # 先验能量 = β * (总邻居数 - 2*相同邻居数)
            prior_energy[k] = self.beta * (total_neighbors - 2 * same_count)
        
        return prior_energy
    
    def icm_update(self, use_vectorized=True):
        """
        执行一次ICM更新
        返回标签变化率
        """
        # 计算对数似然
        log_likelihood = self.compute_log_likelihood()  # [K, H, W]
        
        # 计算先验能量
        prior_energy = self.compute_prior_energy_vectorized()  # [K, H, W]
        
        # 计算总能量
        total_energy = -log_likelihood + prior_energy  # [K, H, W]
        
        # 选择能量最小的类别
        Z_new = torch.argmin(total_energy, dim=0)  # [H, W]
        
        # 计算变化率
        change_rate = torch.mean((Z_new != self.Z_tensor).float()).item()
        
        # 更新标签
        self.Z_tensor = Z_new
        
        return change_rate
    
    def update_parameters(self):
        """
        更新高斯分布参数
        """
        # 将图像展平
        X_flat = self.X_tensor.permute(1, 2, 0).reshape(-1, self.channels)
        Z_flat = self.Z_tensor.view(-1)
        
        # 更新每个类别的参数
        for k in range(self.K):
            mask = Z_flat == k
            count = mask.sum().item()
            
            if count > 0:
                X_k = X_flat[mask]
                
                # 更新均值
                self.means[k] = X_k.mean(dim=0)
                
                # 更新协方差
                if count > 1:
                    diff = X_k - self.means[k].unsqueeze(0)
                    cov_k = torch.mm(diff.t(), diff) / (count - 1)
                    # 添加正则化
                    cov_k = cov_k + 1e-3 * torch.eye(self.channels, device=self.device)
                else:
                    cov_k = torch.eye(self.channels, device=self.device)
                
                self.covariances[k] = cov_k
        
        # 重新预计算逆矩阵和行列式
        self._precompute_inverses()
    
    def segment(self, image_path, update_freq=3):
        """
        执行MRF分割
        
        参数:
        image_path: 图像路径
        update_freq: 参数更新频率（每多少次迭代更新一次参数）
        """
        print("=" * 60)
        print("PyTorch GPU MRF病理图像分割")
        print("=" * 60)
        
        # 记录开始时间
        total_start_time = time.time()
        
        # 1. 加载图像
        print("\n[1/4] 加载图像...")
        start_time = time.time()
        self.load_image(image_path)
        load_time = time.time() - start_time
        print(f"   耗时: {load_time:.2f}秒")
        
        # 2. 初始化标签场
        print("\n[2/4] 初始化标签场...")
        start_time = time.time()
        self.initialize()
        init_time = time.time() - start_time
        print(f"   耗时: {init_time:.2f}秒")
        
        # 3. ICM迭代优化
        print("\n[3/4] ICM迭代优化...")
        
        history = []
        
        for iteration in range(self.max_iter):
            iter_start_time = time.time()
            
            # 执行ICM更新
            change_rate = self.icm_update()
            iter_time = time.time() - iter_start_time
            
            # 定期更新高斯参数
            if update_freq > 0 and (iteration + 1) % update_freq == 0:
                param_start_time = time.time()
                self.update_parameters()
                param_time = time.time() - param_start_time
                
                print(f"   迭代 {iteration+1}/{self.max_iter}: "
                      f"变化率={change_rate:.6f}, "
                      f"时间={iter_time:.3f}s, "
                      f"参数更新={param_time:.3f}s")
            else:
                print(f"   迭代 {iteration+1}/{self.max_iter}: "
                      f"变化率={change_rate:.6f}, "
                      f"时间={iter_time:.3f}s")
            
            # 记录迭代信息
            history.append({
                'iteration': iteration + 1,
                'change_rate': change_rate,
                'time': iter_time
            })
            
            # 检查收敛
            if change_rate < self.tolerance:
                print(f"   收敛于迭代 {iteration+1}")
                break
        
        # 4. 完成
        print("\n[4/4] 分割完成")
        total_time = time.time() - total_start_time
        print(f"   总耗时: {total_time:.2f}秒")
        
        return history
    
    def get_segmentation(self):
        """
        获取分割结果
        """
        return self.Z_tensor.cpu().numpy()
    
    def visualize(self, save_path=None):
        """
        可视化分割结果 - 简化版，只显示3通道合并结果
        """
        # 将数据移回CPU
        image_cpu = self.X_tensor.cpu().permute(1, 2, 0).numpy()
        Z_cpu = self.Z_tensor.cpu().numpy()
        
        # 创建可视化图 - 只显示原始图像、分割结果和边界叠加
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 1. 原始图像
        axes[0].imshow(image_cpu)
        axes[0].set_title('原始病理图像')
        axes[0].axis('off')
        
        # 2. 分割结果 - 使用新的colormap API
        colormap = plt.cm.tab10
        seg_display = colormap(Z_cpu % 10)  # 使用tab10色彩映射
        axes[1].imshow(seg_display)
        axes[1].set_title(f'分割结果 (K={self.K}, β={self.beta})')
        axes[1].axis('off')
        
        # 3. 边界叠加
        overlay = image_cpu.copy()
        
        # 计算边界
        boundaries = np.zeros((self.height, self.width), dtype=bool)
        for k in range(self.K):
            mask = Z_cpu == k
            if np.any(mask):
                # 使用形态学操作找边界
                from scipy import ndimage
                eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3)))
                boundary = mask & ~eroded
                boundaries = boundaries | boundary
        
        # 标记边界为白色
        overlay[boundaries] = [1.0, 1.0, 1.0]
        
        axes[2].imshow(overlay)
        axes[2].set_title('分割边界叠加')
        axes[2].axis('off')
        
        plt.suptitle('MRF病理图像分割结果', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"结果保存到: {save_path}")
        
        plt.show()
    
    def save_results(self, output_dir):
        """
        保存分割结果
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 获取分割结果
        Z_cpu = self.Z_tensor.cpu().numpy()
        
        # 保存为彩色图像 - 使用新的colormap API
        colormap = plt.cm.tab10
        seg_display = (colormap(Z_cpu % 10)[:, :, :3] * 255).astype(np.uint8)
        
        # 保存分割结果
        seg_path = os.path.join(output_dir, 'segmentation.png')
        cv2.imwrite(seg_path, cv2.cvtColor(seg_display, cv2.COLOR_RGB2BGR))
        
        # 保存边界叠加
        image_cpu = self.X_tensor.cpu().permute(1, 2, 0).numpy()
        overlay = image_cpu.copy()
        
        # 计算边界
        boundaries = np.zeros((self.height, self.width), dtype=bool)
        for k in range(self.K):
            mask = Z_cpu == k
            if np.any(mask):
                from scipy import ndimage
                eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3)))
                boundary = mask & ~eroded
                boundaries = boundaries | boundary
        
        # 标记边界为红色
        overlay[boundaries] = [1.0, 0.0, 0.0]
        overlay_path = os.path.join(output_dir, 'boundary_overlay.png')
        cv2.imwrite(overlay_path, (overlay[:, :, ::-1] * 255).astype(np.uint8))
        
        # 保存标签矩阵
        label_path = os.path.join(output_dir, 'labels.npy')
        np.save(label_path, Z_cpu)
        
        print(f"结果保存到目录: {output_dir}")
        print(f"  分割结果: {seg_path}")
        print(f"  边界叠加: {overlay_path}")
        print(f"  标签矩阵: {label_path}")


def main():
    """
    主函数：执行MRF分割
    """
    print("PyTorch GPU MRF病理图像分割")
    print("=" * 60)
    
    # 创建MRF分割器
    mrf = MRFGPUSegmentation(
        K=4,            # 分割类别数
        beta=1.5,       # 耦合系数
        max_iter=30,    # 最大迭代次数
        tolerance=1e-5  # 收敛阈值
    )
    
    # 执行分割
    image_path = "PGMProject2025-data/Task2-Visium_HD-Crop.png"
    
    history = mrf.segment(
        image_path=image_path,
        update_freq=3  # 每3次迭代更新一次参数
    )
    
    # 可视化结果
    mrf.visualize(save_path='mrf_segmentation_result.png')
    
    # 保存结果
    mrf.save_results(output_dir='segmentation_results')
    
    # 显示分割统计信息
    Z = mrf.get_segmentation()
    unique, counts = np.unique(Z, return_counts=True)
    
    print("\n分割统计:")
    print("-" * 30)
    for label, count in zip(unique, counts):
        percentage = count / Z.size * 100
        print(f"类别 {label}: {count:,} 像素 ({percentage:.1f}%)")
    
    # 绘制收敛曲线
    if history:
        iterations = [h['iteration'] for h in history]
        change_rates = [h['change_rate'] for h in history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, change_rates, 'b-o', linewidth=2, markersize=8)
        plt.xlabel('迭代次数', fontsize=12)
        plt.ylabel('标签变化率', fontsize=12)
        plt.title('ICM迭代收敛曲线', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig('convergence_curve.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    print("\n" + "=" * 60)
    print("分割完成！")
    print("=" * 60)


if __name__ == "__main__":
    # 运行主函数
    main()