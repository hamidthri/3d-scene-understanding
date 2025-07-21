import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans


class SemanticSimilarityModule(nn.Module):
    def __init__(self, feature_dim=64, temperature=0.1):
        super(SemanticSimilarityModule, self).__init__()
        self.feature_dim = feature_dim
        self.temperature = temperature
        
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(3, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, feature_dim, 1)
        )
        
    def forward(self, xyz):
        B, N, _ = xyz.shape
        xyz_t = xyz.permute(0, 2, 1)
        
        features = self.feature_extractor(xyz_t)
        features = features.permute(0, 2, 1)
        features = F.normalize(features, dim=-1)
        
        similarity_matrix = torch.bmm(features, features.transpose(1, 2))
        similarity_matrix = similarity_matrix / self.temperature
        
        return similarity_matrix, features


class ContrastiveSamplingLoss(nn.Module):
    def __init__(self, pos_threshold=0.7, neg_threshold=0.3, margin=0.2):
        super(ContrastiveSamplingLoss, self).__init__()
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        self.margin = margin
        
    def forward(self, sampling_probs, similarity_matrix):
        B, N = sampling_probs.shape
        
        pos_mask = similarity_matrix > self.pos_threshold
        neg_mask = similarity_matrix < self.neg_threshold
        
        prob_diff = torch.abs(sampling_probs.unsqueeze(2) - sampling_probs.unsqueeze(1))
        
        pos_loss = torch.sum(prob_diff * pos_mask.float()) / (torch.sum(pos_mask.float()) + 1e-8)
        neg_loss = torch.clamp(self.margin - prob_diff, min=0)
        neg_loss = torch.sum(neg_loss * neg_mask.float()) / (torch.sum(neg_mask.float()) + 1e-8)
        
        return pos_loss + neg_loss


class SemanticClusteringModule(nn.Module):
    def __init__(self, k_neighbors=16, min_cluster_size=8):
        super(SemanticClusteringModule, self).__init__()
        self.k_neighbors = k_neighbors
        self.min_cluster_size = min_cluster_size
        
    def compute_semantic_knn(self, features, k):
        B, N, D = features.shape
        
        similarity = torch.bmm(features, features.transpose(1, 2))
        _, knn_indices = torch.topk(similarity, k=k+1, dim=-1)
        knn_indices = knn_indices[:, :, 1:]
        
        return knn_indices
    
    def form_semantic_clusters(self, features, knn_indices):
        B, N, D = features.shape
        _, _, K = knn_indices.shape
        
        clusters = []
        for b in range(B):
            visited = torch.zeros(N, dtype=torch.bool, device=features.device)
            batch_clusters = []
            
            for i in range(N):
                if visited[i]:
                    continue
                
                cluster = [i]
                queue = [i]
                visited[i] = True
                
                while queue:
                    current = queue.pop(0)
                    neighbors = knn_indices[b, current]
                    
                    for neighbor in neighbors:
                        if not visited[neighbor]:
                            cluster.append(neighbor.item())
                            queue.append(neighbor.item())
                            visited[neighbor] = True
                            
                            if len(cluster) >= self.min_cluster_size:
                                break
                    
                    if len(cluster) >= self.min_cluster_size:
                        break
                
                if len(cluster) >= 3:
                    batch_clusters.append(cluster)
            
            clusters.append(batch_clusters)
        
        return clusters
    
    def forward(self, features):
        knn_indices = self.compute_semantic_knn(features, self.k_neighbors)
        clusters = self.form_semantic_clusters(features, knn_indices)
        return clusters, knn_indices


class CrossInstanceConsistency(nn.Module):
    def __init__(self, feature_dim=64, num_prototypes=10):
        super(CrossInstanceConsistency, self).__init__()
        self.feature_dim = feature_dim
        self.num_prototypes = num_prototypes
        
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, feature_dim))
        self.prototype_mlp = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_prototypes)
        )
        
    def compute_prototype_assignment(self, features):
        B, N, D = features.shape
        
        prototype_scores = self.prototype_mlp(features)
        prototype_assignment = F.softmax(prototype_scores, dim=-1)
        
        return prototype_assignment
    
    def compute_consistency_loss(self, sampling_probs, prototype_assignment):
        B, N, P = prototype_assignment.shape
        
        weighted_probs = torch.sum(sampling_probs.unsqueeze(-1) * prototype_assignment, dim=1)
        
        consistency_loss = 0
        for p in range(P):
            prototype_probs = weighted_probs[:, p]
            mean_prob = torch.mean(prototype_probs)
            consistency_loss += torch.mean((prototype_probs - mean_prob) ** 2)
        
        return consistency_loss / P
    
    def forward(self, features, sampling_probs):
        prototype_assignment = self.compute_prototype_assignment(features)
        consistency_loss = self.compute_consistency_loss(sampling_probs, prototype_assignment)
        
        return consistency_loss, prototype_assignment


class ContrastiveSamplingNetwork(nn.Module):
    def __init__(self, input_dim=3, feature_dim=64, temperature=0.1, k_neighbors=16):
        super(ContrastiveSamplingNetwork, self).__init__()
        
        self.semantic_similarity = SemanticSimilarityModule(feature_dim, temperature)
        self.clustering_module = SemanticClusteringModule(k_neighbors)
        self.consistency_module = CrossInstanceConsistency(feature_dim)
        
        self.sampling_network = nn.Sequential(
            nn.Conv1d(feature_dim + 3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 1, 1),
            nn.Sigmoid()
        )
        
        self.contrastive_loss = ContrastiveSamplingLoss()
        
    def forward(self, xyz, labels=None):
        B, N, _ = xyz.shape
        
        similarity_matrix, semantic_features = self.semantic_similarity(xyz)
        
        combined_features = torch.cat([xyz, semantic_features], dim=-1)
        combined_features = combined_features.permute(0, 2, 1)
        
        sampling_probs = self.sampling_network(combined_features).squeeze(1)
        
        clusters, knn_indices = self.clustering_module(semantic_features)
        
        contrastive_loss = self.contrastive_loss(sampling_probs, similarity_matrix)
        
        consistency_loss = 0
        if labels is not None:
            consistency_loss, _ = self.consistency_module(semantic_features, sampling_probs)
        
        return sampling_probs, contrastive_loss, consistency_loss, clusters, semantic_features
    
    def cluster_aware_sampling(self, xyz, sampling_probs, clusters, npoint):
        B, N, _ = xyz.shape
        sampled_indices = []
        
        for b in range(B):
            batch_clusters = clusters[b]
            batch_probs = sampling_probs[b]
            
            cluster_scores = []
            for cluster in batch_clusters:
                cluster_tensor = torch.tensor(cluster, device=xyz.device)
                cluster_prob = torch.mean(batch_probs[cluster_tensor])
                cluster_scores.append((cluster_prob, cluster))
            
            cluster_scores.sort(key=lambda x: x[0], reverse=True)
            
            selected_indices = []
            points_needed = npoint
            
            for score, cluster in cluster_scores:
                if points_needed <= 0:
                    break
                
                cluster_size = min(len(cluster), points_needed)
                cluster_indices = np.random.choice(cluster, cluster_size, replace=False)
                selected_indices.extend(cluster_indices)
                points_needed -= cluster_size
            
            if len(selected_indices) < npoint:
                remaining_indices = list(set(range(N)) - set(selected_indices))
                additional_needed = npoint - len(selected_indices)
                if len(remaining_indices) >= additional_needed:
                    additional_indices = np.random.choice(remaining_indices, additional_needed, replace=False)
                    selected_indices.extend(additional_indices)
                else:
                    selected_indices.extend(remaining_indices)
            
            selected_indices = selected_indices[:npoint]
            sampled_indices.append(selected_indices)
        
        return torch.tensor(sampled_indices, device=xyz.device, dtype=torch.long)