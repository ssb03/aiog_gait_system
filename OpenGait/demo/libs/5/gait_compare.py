import torch
import numpy as np

def getemb(data):
    return data["inference_feat"]

# def computedistence(x, y):
#     distance = torch.sqrt(torch.sum(torch.square(x - y)))
#     return distance

def computedistence(x, y):
    print(f"x shape before flatten: {x.shape}")
    print(f"y shape before flatten: {y.shape}")
    if x.ndim == 3:
        x = x.mean(dim=0)  # Mean over time (frame) dimension
    if y.ndim == 3:
        y = y.mean(dim=0)
    x_norm = x / torch.norm(x)
    y_norm = y / torch.norm(y)
    return 1 - torch.dot(x_norm, y_norm)

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
                    # distance = computedistence(embs, gallery_emb)
                    probe_vec = embs.mean(dim=1).flatten()
                    gallery_vec = gallery_emb.mean(dim=1).flatten()
                    distance = computedistence(probe_vec, gallery_vec)
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
