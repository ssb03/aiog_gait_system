import torch
import numpy as np
import torch.nn.functional as F

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

def comparefeat(embs, gallery_feat: dict, pid, threshold_value=0.02): 
    print(f"[DEBUG] Using threshold: {threshold_value}")

    probe_name = pid.split("-")[0]
    min_distance = float("inf")
    matched_id = None
    distances = {}

    for gallery_id, subject_dict in gallery_feat.items():
        # if gallery_id == probe_name:
        #     continue
        for subject_id, type_dict in subject_dict.items():
            for view, gallery_emb in type_dict.items():
                gid = f"{gallery_id}-{subject_id}"
                try:
                    distance = computedistence(embs, gallery_emb)
                    # probe_vec = embs.mean(dim=1).flatten()
                    # gallery_vec = gallery_emb.mean(dim=1).flatten()
                    # distance = computedistence(probe_vec, gallery_vec)
                except Exception as e:
                    print(f"[ERROR] Computing distance failed for {gid}: {e}")
                    continue

                distances[gid] = distance
                print(f"Distance: {pid} vs {gid} = {distance:.4f}")  # Debug log

                if distance < min_distance:
                    matched_id = gid
                    min_distance = distance

    print(f"[DEBUG] Final min_distance: {min_distance}, threshold_value: {threshold_value}, passed: {min_distance <= threshold_value}")

    if min_distance <= threshold_value:
        print(f"✅ Match: {pid} → {matched_id} (distance={min_distance:.4f})")
        return matched_id, sorted(distances.items(), key=lambda x: x[1])
    else:
        print(f"❌ [NO MATCH] {pid} has no match under threshold {threshold_value}. Closest: {matched_id} @ {min_distance:.4f}")
        return "no_match", sorted(distances.items(), key=lambda x: x[1])
