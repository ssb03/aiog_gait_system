import torch
import numpy as np
import torch.nn.functional as F

class GaitComparator:
    def __init__(self, gallery_features):
        self.cov_inv = self._compute_covariance(gallery_features)
        self.threshold = self._compute_threshold(gallery_features)
        
    def _compute_covariance(self, gallery_features):
        """Compute inverse covariance matrix for Mahalanobis distance"""
        all_features = []
        for subject in gallery_features.values():
            for view_feat in subject.values():
                all_features.append(view_feat.squeeze(0).cpu().numpy())
        cov_matrix = np.cov(np.vstack(all_features).T)
        return torch.inverse(torch.tensor(cov_matrix, dtype=torch.float32).cuda())

    def _compute_threshold(self, gallery_features):
        """Compute 95th percentile of inter-class distances"""
        distances = []
        keys = list(gallery_features.keys())
        for i in range(len(keys)):
            for j in range(i+1, len(keys)):
                dist = self.computedistence(
                    gallery_features[keys[i]]['undefined'],
                    gallery_features[keys[j]]['undefined']
                )
                distances.append(dist)
        return np.percentile(distances, 95)
    
    def computedistence(self, x, y):
        """Improved distance metric combining temporal and spatial features"""
        x = x.squeeze(0)  # [num_frames, feature_dim]
        y = y.squeeze(0)
        
        # Temporal attention
        x_weights = F.softmax(torch.norm(x, dim=1), dim=0)
        y_weights = F.softmax(torch.norm(y, dim=1), dim=0)
        x_mean = (x * x_weights.unsqueeze(1)).sum(dim=0)
        y_mean = (y * y_weights.unsqueeze(1)).sum(dim=0)
        
        # Mahalanobis distance
        diff = x_mean - y_mean
        return torch.sqrt(diff @ self.cov_inv @ diff.T).item()

def getemb(data):
    return data["inference_feat"]

# DTW
from fastdtw import fastdtw
def computedistence(x, y):
    distance, _ = fastdtw(x.cpu().numpy(), y.cpu().numpy())
    return torch.tensor(distance)

#Default
# def computedistence(x, y):
#     distance = torch.sqrt(torch.sum(torch.square(x - y)))
#     return distance

# def computedistence(x, y):
#     print(f"x shape before flatten: {x.shape}")
#     print(f"y shape before flatten: {y.shape}")
#     if x.ndim == 3:
#         x = x.mean(dim=0)  # Mean over time (frame) dimension
#     if y.ndim == 3:
#         y = y.mean(dim=0)
#     x_norm = x / torch.norm(x)
#     y_norm = y / torch.norm(y)
#     return 1 - torch.dot(x_norm, y_norm)

# def computedistence(x, y):
#     print(f"x shape before processing: {x.shape}")
#     print(f"y shape before processing: {y.shape}")
    
#     # Normalize along the channel dimension
#     x_norm = F.normalize(x, p=2, dim=1)
#     y_norm = F.normalize(y, p=2, dim=1)
    
#     # Compute cosine similarity directly on 3D features
#     distance = 1 - torch.einsum('nct,mct->nm', x_norm, y_norm)
#     return distance



# def wasserstein_distance(x, y):
#     x_flat = x.flatten()
#     y_flat = y.flatten()
    
#     # Sort the samples
#     x_sorted = torch.sort(x_flat)[0]
#     y_sorted = torch.sort(y_flat)[0]
    
#     # Compute the distance
#     return torch.mean(torch.abs(x_sorted - y_sorted))
# # from torchmetrics.image import WassersteinDistance
# def computedistence(x, y):
#     if x.ndim > 1:
#         x = x.flatten()
#     if y.ndim > 1:
#         y = y.flatten()
    
#     # Normalize the vectors
#     x_norm = F.normalize(x, p=2, dim=0)
#     y_norm = F.normalize(y, p=2, dim=0)
    
#     # Compute Wasserstein distance
#     return wasserstein_distance(x_norm, y_norm)

# def computedistence(x, y):
#     """Compute cosine distance between two gait feature tensors"""
#     # x and y shapes: [1, num_frames, feature_dim] or [num_frames, feature_dim]
    
#     # Handle batch dimension
#     if x.dim() == 3:
#         x = x.squeeze(0)  # Remove batch dim if present
#     if y.dim() == 3:
#         y = y.squeeze(0)
    
#     # Normalize along feature dimension
#     x_norm = F.normalize(x, p=2, dim=1)
#     y_norm = F.normalize(y, p=2, dim=1)
    
#     # Compute mean cosine distance across frames
#     cos_dist = 1 - torch.einsum('nf,mf->nm', x_norm, y_norm).mean()
    
#     return cos_dist

# def computedistence(x, y, cov_inv):
#     diff = x - y
#     return torch.sqrt(diff.T @ cov_inv @ diff)


def compareid(data, dict, pid, threshold_value):
    probe_name = pid.split("-")[0]
    embs = getemb(data)
    min = threshold_value
    id = None
    dic={}
    for key in dict:
        if key == probe_name:
            continue
        for subject in dict[key]:
            for type in subject:
                for view in subject[type]:
                    value = subject[type][view]
                    distance = computedistence(embs["embeddings"],value)
                    gid = key + "-" + str(type)
                    gid_distance = (gid, distance)
                    dic[gid] = distance
                    if distance.float() < min:
                        id = gid
                        min = distance.float()
    dic_sort= sorted(dic.items(), key=lambda d:d[1], reverse = False)
    if id is None:
        print("############## no id #####################")
    return id, dic_sort


# def comparefeat(embs, gallery_feat: dict, pid, threshold_value):
#     """Compares the distance between features

#     Args:
#         embs (Tensor): Embeddings of person with pid
#         gallery_feat (dict): Dictionary of features from gallery
#         pid (str): The id of person in probe
#         threshold_value (int): Threshold
#     Returns:
#         id (str): The id in gallery
#         dic_sort (dict): Recognition result sorting dictionary
#     """
#     probe_name = pid.split("-")[0]
#     min = threshold_value
#     id = None
#     dic={}
#     for key in gallery_feat:
#         if key == probe_name:
#             continue
#         for subject in gallery_feat[key]:
#             for type in subject:
#                 for view in subject[type]:
#                     value = subject[type][view]
#                     distance = computedistence(embs, value)
#                     gid = key + "-" + str(type)
#                     gid_distance = (gid, distance)
#                     dic[gid] = distance
#                     if distance.float() < min:
#                         id = gid
#                         min = distance.float()
#     dic_sort= sorted(dic.items(), key=lambda d:d[1], reverse = False)
#     if id is None:
#         print("############## no id #####################")
#     return id, dic_sort

def comparefeat(embs, gallery_feat, pid, comparator):
    """Updated comparison function using the GaitComparator"""
    probe_name = pid.split("-")[0]
    min_distance = float("inf")
    matched_id = None
    distances = {}

    for gallery_id, subject_feat in gallery_feat.items():
        for view, gallery_emb in subject_feat.items():
            gid = f"{gallery_id}-{view}"
            try:
                distance = comparator.computedistence(embs['undefined'], gallery_emb)
                distances[gid] = distance
                
                if distance < min_distance:
                    matched_id = gid
                    min_distance = distance
            except Exception as e:
                print(f"Error comparing {pid} vs {gid}: {e}")

    print(f"Best match for {pid}: {matched_id} (distance={min_distance:.4f})")
    return matched_id, sorted(distances.items(), key=lambda x: x[1])
