"""异常检测器 - 改进版

核心改进：
1. PCA保留更多维度（不低于min_components），避免丢失缺陷信号
2. 分数归一化后融合，避免量纲问题
3. 支持FNR优先的阈值策略（宁可误杀不漏杀）
4. 增大记忆库
5. 多种评分策略可选
"""
import numpy as np
import torch
from tqdm import tqdm


class AnomalyDetector:
    """
    改进版异常检测器
    融合PCA、超球面距离和记忆库方法
    """
    def __init__(self, device='cpu', n_components=None, use_hypersphere=True, 
                 use_memory_bank=True, memory_ratio=0.05,
                 min_pca_components=32, pca_variance=0.995,
                 score_mode='combined'):
        """
        Args:
            min_pca_components: PCA最少保留维度（之前9维太少）
            pca_variance: PCA保留方差比例（从0.95提到0.995）
            memory_ratio: 记忆库比例（从0.01提到0.05）
            score_mode: 评分模式
                - 'combined': 归一化融合（默认，推荐）
                - 'mahal': 仅马氏距离
                - 'memory': 仅记忆库距离
                - 'max': 取各分数最大值
        """
        self.device = device
        self.n_components = n_components
        self.use_hypersphere = use_hypersphere
        self.use_memory_bank = use_memory_bank
        self.memory_ratio = memory_ratio
        self.min_pca_components = min_pca_components
        self.pca_variance = pca_variance
        self.score_mode = score_mode
        
        self.mean = None
        self.cov_inv = None
        self.pca_mean = None
        self.pca_components = None
        self.pca_variances = None  # 各主成分方差（用于归一化马氏距离）
        self.hypersphere_center = None
        self.hypersphere_radius = None
        self.memory_bank = None
        # 归一化统计量
        self.score_stats = {}  # 各分数分量的mean/std，用于归一化融合

    def fit(self, encoder, dataloader, use_multiscale=True):
        """用正常样本拟合分布"""
        encoder.eval()
        features = []

        with torch.no_grad():
            for imgs, labels, _ in tqdm(dataloader, desc="提取特征"):
                normal_mask = (labels == 0)
                if normal_mask.sum() == 0:
                    continue
                imgs_normal = imgs[normal_mask].to(self.device)
                
                if hasattr(encoder, 'forward_multiscale') and use_multiscale:
                    feat = encoder.forward_multiscale(imgs_normal)
                else:
                    feat = encoder.forward_features(imgs_normal)
                features.append(feat.cpu().numpy())

        features = np.concatenate(features, axis=0)
        print(f"[INFO] 正常样本特征: {features.shape}")

        # PCA降维 - 保留更多维度
        n_samples, n_features = features.shape
        
        if self.n_components is None:
            self.n_components = min(n_features, n_samples // 2, 128)
        
        self.pca_mean = np.mean(features, axis=0)
        features_centered = features - self.pca_mean
        
        U, S, Vt = np.linalg.svd(features_centered, full_matrices=False)
        var_explained = (S ** 2) / np.sum(S ** 2)
        cumulative_var = np.cumsum(var_explained)
        
        # 按方差比例确定维度，但不少于min_pca_components
        n_var = np.searchsorted(cumulative_var, self.pca_variance) + 1
        n_components = max(n_var, self.min_pca_components)
        n_components = min(n_components, self.n_components, n_features)
        
        self.pca_components = Vt[:n_components, :]
        self.pca_variances = S[:n_components] ** 2 / n_samples  # 各主成分方差
        features_pca = features_centered @ self.pca_components.T
        
        print(f"[INFO] PCA: {n_features}D -> {n_components}D "
              f"(保留{cumulative_var[n_components-1]*100:.1f}%方差, "
              f"方差阈值={self.pca_variance*100:.1f}%, 最低维度={self.min_pca_components})")

        # 计算马氏距离参数
        self.mean = np.mean(features_pca, axis=0)
        cov, shrinkage = self._ledoit_wolf_shrinkage(features_pca)
        print(f"[INFO] Ledoit-Wolf收缩系数: {shrinkage:.4f}")
        
        cov += np.eye(cov.shape[0]) * 1e-6

        try:
            self.cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            print("[WARNING] 协方差矩阵奇异，使用伪逆")
            self.cov_inv = np.linalg.pinv(cov)

        # 超球面参数
        if self.use_hypersphere:
            self.hypersphere_center = self.pca_mean
            distances = np.linalg.norm(features - self.hypersphere_center, axis=1)
            self.hypersphere_radius = np.percentile(distances, 95)
            print(f"[INFO] 超球面半径: {self.hypersphere_radius:.4f}")

        # 记忆库（增大采样比例）
        if self.use_memory_bank:
            n_memory = max(1, int(n_samples * self.memory_ratio))
            # 用k-means++风格的核心集选择：覆盖更均匀
            self.memory_bank = self._select_coreset(features, n_memory)
            print(f"[INFO] 记忆库大小: {n_memory} (比例: {self.memory_ratio})")

        # 计算正常样本自身分数分布（用于归一化）
        self._compute_score_stats(features)

        print("[INFO] 异常检测器拟合完成")

    def _select_coreset(self, features, n_memory):
        """核心集选择：最大化覆盖"""
        n_samples = features.shape[0]
        if n_memory >= n_samples:
            return features.copy()
        
        # 随机选第一个点
        rng = np.random.RandomState(42)
        indices = [rng.randint(n_samples)]
        
        # 贪心选最远点
        for _ in range(n_memory - 1):
            min_dists = np.full(n_samples, np.inf)
            for idx in indices:
                dists = np.linalg.norm(features - features[idx], axis=1)
                min_dists = np.minimum(min_dists, dists)
            # 概率正比于距离，加入随机性
            probs = min_dists / (min_dists.sum() + 1e-8)
            new_idx = rng.choice(n_samples, p=probs)
            indices.append(new_idx)
        
        return features[indices]

    def _compute_score_stats(self, features):
        """计算正常样本各分数分量的统计量（用于归一化融合）"""
        n = features.shape[0]
        # 采样计算，避免太慢
        sample_size = min(500, n)
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(n, sample_size, replace=False)
        sample_feats = features[sample_idx]
        
        mahal_scores = []
        sphere_scores = []
        memory_scores = []
        
        for i in range(sample_size):
            feat_pca = (sample_feats[i] - self.pca_mean) @ self.pca_components.T
            
            # 马氏距离
            diff = feat_pca - self.mean
            mahal_dist = np.sqrt(diff @ self.cov_inv @ diff)
            mahal_scores.append(mahal_dist)
            
            # 超球面
            if self.use_hypersphere:
                raw_dist = np.linalg.norm(sample_feats[i] - self.hypersphere_center)
                sphere_dist = max(0, raw_dist - self.hypersphere_radius)
                sphere_scores.append(sphere_dist)
            
            # 记忆库
            if self.use_memory_bank and self.memory_bank is not None:
                dists = np.linalg.norm(self.memory_bank - sample_feats[i], axis=1)
                memory_scores.append(np.min(dists))
        
        self.score_stats['mahal'] = {
            'mean': np.mean(mahal_scores),
            'std': np.std(mahal_scores) + 1e-8
        }
        if sphere_scores:
            self.score_stats['sphere'] = {
                'mean': np.mean(sphere_scores),
                'std': np.std(sphere_scores) + 1e-8
            }
        if memory_scores:
            self.score_stats['memory'] = {
                'mean': np.mean(memory_scores),
                'std': np.std(memory_scores) + 1e-8
            }

    def _ledoit_wolf_shrinkage(self, X):
        """Ledoit-Wolf收缩估计"""
        n_samples, n_features = X.shape
        
        sample_cov = np.cov(X, rowvar=False)
        target = np.diag(np.diag(sample_cov))
        
        mu = np.trace(sample_cov) / n_features
        delta = np.sum((sample_cov - target) ** 2) / n_features
        
        X_centered = X - X.mean(axis=0)
        beta = 0
        for i in range(n_samples):
            outer_i = np.outer(X_centered[i], X_centered[i])
            beta += np.sum((outer_i - sample_cov) ** 2)
        beta = beta / (n_samples ** 2)
        
        shrinkage = min(1, max(0, beta / delta)) if delta > 0 else 1
        shrunk_cov = shrinkage * target + (1 - shrinkage) * sample_cov
        
        return shrunk_cov, shrinkage

    def _compute_single_score(self, feat_raw, feat_pca):
        """计算单个样本的各分数分量"""
        # 1. 马氏距离
        diff = feat_pca - self.mean
        mahal_dist = np.sqrt(diff @ self.cov_inv @ diff)
        
        # 2. 超球面距离
        sphere_dist = 0
        if self.use_hypersphere:
            raw_dist = np.linalg.norm(feat_raw - self.hypersphere_center)
            sphere_dist = max(0, raw_dist - self.hypersphere_radius)
        
        # 3. 记忆库距离
        memory_dist = 0
        if self.use_memory_bank and self.memory_bank is not None:
            dists = np.linalg.norm(self.memory_bank - feat_raw, axis=1)
            memory_dist = np.min(dists)
        
        return mahal_dist, sphere_dist, memory_dist

    def _combine_scores(self, mahal, sphere, memory):
        """融合分数（归一化后加权）"""
        if self.score_mode == 'mahal':
            return mahal
        elif self.score_mode == 'memory':
            return memory if memory > 0 else mahal
        elif self.score_mode == 'max':
            scores = [mahal]
            if sphere > 0:
                scores.append(sphere)
            if memory > 0:
                scores.append(memory)
            return max(scores)
        
        # combined: 归一化后加权融合
        normalized = []
        weights = []
        
        # 马氏距离（权重最高，最可靠）
        if 'mahal' in self.score_stats:
            norm_mahal = (mahal - self.score_stats['mahal']['mean']) / self.score_stats['mahal']['std']
            normalized.append(max(0, norm_mahal))  # 低于均值的不算异常
            weights.append(1.0)
        
        # 超球面
        if sphere > 0 and 'sphere' in self.score_stats:
            norm_sphere = (sphere - self.score_stats['sphere']['mean']) / self.score_stats['sphere']['std']
            normalized.append(max(0, norm_sphere))
            weights.append(0.3)
        
        # 记忆库
        if memory > 0 and 'memory' in self.score_stats:
            norm_memory = (memory - self.score_stats['memory']['mean']) / self.score_stats['memory']['std']
            normalized.append(max(0, norm_memory))
            weights.append(0.5)
        
        if not normalized:
            return mahal
        
        # 加权平均
        total_weight = sum(weights)
        combined = sum(n * w for n, w in zip(normalized, weights)) / total_weight
        return combined

    def score(self, encoder, dataloader, use_multiscale=True):
        """计算异常分数"""
        encoder.eval()
        all_scores = []
        all_labels = []
        all_paths = []

        with torch.no_grad():
            for imgs, labels, paths in tqdm(dataloader, desc="检测异常"):
                imgs = imgs.to(self.device)
                
                if hasattr(encoder, 'forward_multiscale') and use_multiscale:
                    feat = encoder.forward_multiscale(imgs).cpu().numpy()
                else:
                    feat = encoder.forward_features(imgs).cpu().numpy()

                # PCA降维
                feat_pca = (feat - self.pca_mean) @ self.pca_components.T

                for i in range(feat_pca.shape[0]):
                    mahal, sphere, memory = self._compute_single_score(feat[i], feat_pca[i])
                    combined_score = self._combine_scores(mahal, sphere, memory)
                    
                    all_scores.append(combined_score)
                    all_labels.append(labels[i].item())
                    all_paths.append(paths[i])

        return np.array(all_scores), np.array(all_labels), all_paths

    def find_threshold_fnr_priority(self, scores, labels, target_fnr=0.0):
        """
        找到满足目标漏检率(FNR)的最低阈值
        宁可FP高也要把FN压低
        
        Args:
            target_fnr: 目标漏检率（默认0.0，即不漏检任何一个缺陷）
        
        Returns:
            threshold: 满足FNR约束的阈值
        """
        defect_scores = scores[labels == 1]
        normal_scores = scores[labels == 0]
        
        if len(defect_scores) == 0:
            print("[WARNING] 没有缺陷样本，无法确定阈值")
            return np.median(scores)
        
        # 阈值设为：所有缺陷样本分数的最小值（确保0漏检）
        # 再适当下调一点留安全余量
        if target_fnr == 0.0:
            # 0漏检：阈值 = min(缺陷分数) * 衰减系数
            min_defect_score = np.min(defect_scores)
            threshold = min_defect_score * 0.95  # 留5%余量
        else:
            # 允许一定漏检率
            sorted_defect = np.sort(defect_scores)
            n_allow_miss = int(len(defect_scores) * target_fnr)
            if n_allow_miss < len(defect_scores):
                threshold = sorted_defect[n_allow_miss]
            else:
                threshold = np.min(normal_scores)
        
        # 计算该阈值下的各项指标
        preds = (scores > threshold).astype(int)
        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        tn = ((preds == 0) & (labels == 0)).sum()
        
        actual_fnr = fn / (tp + fn + 1e-8)
        actual_fpr = fp / (fp + tn + 1e-8)
        
        print(f"[INFO] FNR优先阈值: {threshold:.4f}")
        print(f"  -> 漏检率(FNR): {actual_fnr:.4f}, 误检率(FPR): {actual_fpr:.4f}")
        print(f"  -> TP={tp}, FP={fp}, FN={fn}, TN={tn}")
        
        return threshold
