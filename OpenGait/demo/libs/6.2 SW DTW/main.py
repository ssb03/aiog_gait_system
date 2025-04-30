import os
import os.path as osp
import time
import sys
import pickle

sys.path.append(os.path.abspath('.') + "/demo/libs/")
from track import *
from segment import *
from recognise import *

# -------- Utility Functions --------

def load_features(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_features(features, path):
    with open(path, 'wb') as f:
        pickle.dump(features, f)

def load_from_pickle(path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

def save_to_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def get_video_paths(root_dir):
    video_paths = []
    for subject_id in os.listdir(root_dir):
        subject_dir = osp.join(root_dir, subject_id)
        if not osp.isdir(subject_dir):
            continue
        for view_id in os.listdir(subject_dir):
            view_dir = osp.join(subject_dir, view_id)
            if not osp.isdir(view_dir):
                continue
            for video_file in os.listdir(view_dir):
                if video_file.endswith(".mp4"):
                    full_path = osp.join(view_dir, video_file)
                    video_paths.append(full_path)
    return video_paths
# -------- Main Function --------

def main():
    base_output = "./demo/output_pipeline/"
    video_output_dir = osp.join(base_output, "videos/")
    silhouette_root = osp.join(base_output, "silhouettes/")
    feature_root = osp.join(base_output, "features/")
    track_root = osp.join(base_output, "tracks/")

    os.makedirs(video_output_dir, exist_ok=True)
    os.makedirs(silhouette_root, exist_ok=True)
    os.makedirs(feature_root, exist_ok=True)
    os.makedirs(track_root, exist_ok=True)

    # gallery_video_paths = get_video_paths("./demo/data/Gallery")
    # probe_video_paths = get_video_paths("./demo/data/Probe")

    all_paths = [
        "./demo/data/001/000/1.mp4",
        "./demo/data/001/090/1.mp4",
        "./demo/data/001/180/1.mp4",

        "./demo/data/002/000/1.mp4",

        "./demo/data/002/090/1.mp4",
        "./demo/data/002/090/2.mp4",

        "./demo/data/002/180/1.mp4",
        "./demo/data/002/180/2.mp4",

        "./demo/data/003/090/1.mp4",

        "./demo/data/004/000/1.mp4",
        "./demo/data/004/000/2.mp4",
        "./demo/data/004/090/1.mp4",
        "./demo/data/004/090/2.mp4",
        "./demo/data/004/180/1.mp4",
        "./demo/data/004/180/2.mp4",

        "./demo/data/005/090/1.mp4",
        "./demo/data/005/090/2.mp4",
        "./demo/data/005/180/1.mp4",

        "./demo/data/006/180/1.mp4",
    ]

    gallery_video_paths = [
        "./demo/data/001/000/1.mp4",
        "./demo/data/002/090/1.mp4",
        "./demo/data/002/180/1.mp4",
        "./demo/data/004/000/1.mp4",
        "./demo/data/004/090/1.mp4",
        "./demo/data/004/180/1.mp4",
        "./demo/data/005/090/1.mp4",
        # "./demo/data/005/180/1.mp4",

    ]

    probe_video_paths = [
        "./demo/data/001/000/1.mp4",
        "./demo/data/001/180/1.mp4",
        "./demo/data/002/090/2.mp4",
        "./demo/data/002/180/2.mp4",
        "./demo/data/003/090/1.mp4",
        "./demo/data/004/000/2.mp4",
        "./demo/data/004/090/2.mp4",
        "./demo/data/004/180/2.mp4",
        "./demo/data/005/090/2.mp4",
        "./demo/data/006/180/1.mp4",
    ]

    gallery_features = {}  # Dictionary to store gallery features (person_id: features)
    gallery_track_results = {} # Dictionary to store track results for galleries
    gallery_silhouettes = {} # Dictionary to store silhouette paths for galleries

    print("Processing Gallery Videos...")
    for gallery_video_path in gallery_video_paths:
        
        video_name_base = gallery_video_path.split("/")[-1].split(".")[0]
        person_id = video_name_base.split("-")[0]  # Extract person ID from video name

        path_parts = os.path.normpath(gallery_video_path).split(os.sep)
        subject_id = path_parts[-3]  # e.g. "001"
        view_id = path_parts[-2]     # e.g. "090"
        seq_id = os.path.splitext(path_parts[-1])[0]  # e.g. "01"
        print("Path parts:", path_parts)
        print("Subject ID:", subject_id)
        print("View ID:", view_id)
        print("Sequence ID:", seq_id)
        video_name_base = f"{subject_id}-{view_id}-{seq_id}"

        video_save_folder = osp.join(video_output_dir, f"gallery_{video_name_base}")
        os.makedirs(video_save_folder, exist_ok=True)

        print(f"  Processing: {gallery_video_path}")

        track_path = osp.join(track_root, f"gallery_{video_name_base}_track.pkl")
        track_result = load_from_pickle(track_path)
        if track_result is None:
            track_result = track(gallery_video_path, video_save_folder)
            save_to_pickle(track_result, track_path)
        else:
            print(f"    [CACHE] Loaded tracking result: {track_path}")


        silhouette_folder = osp.join(silhouette_root, 'GaitSilhouette', f"gallery_{video_name_base}")
        os.makedirs(silhouette_folder, exist_ok=True)
        gallery_silhouette_path = osp.join(silhouette_folder, f"{video_name_base}_silhouette.pkl") # Save as pickle for multiple individuals

        # Check if silhouette exists, if so, load (you might need to adapt getsil for multiple people)
        if os.path.exists(gallery_silhouette_path):
            print(f"    Loading existing silhouette: {gallery_silhouette_path}")
            # Assuming getsil can load multiple silhouettes from a saved file
            gallery_silhouettes[gallery_video_path] = getsil(gallery_video_path, silhouette_folder)
            
        else:
            print("    Segmenting silhouettes...")
            gallery_silhouettes[gallery_video_path] = seg(gallery_video_path, track_result, silhouette_folder)
            save_features(gallery_silhouettes[gallery_video_path], gallery_silhouette_path)
            # Save the generated silhouettes for potential reuse (adapt to handle multiple people)
            # You might need a function to save the segmented outputs
            # For simplicity, we'll assume 'seg' saves them appropriately in the folder

        feature_folder = osp.join(feature_root, 'GaitFeatures', f"gallery_{video_name_base}")
        os.makedirs(feature_folder, exist_ok=True)
        gallery_feature_path = osp.join(feature_folder, f"{video_name_base}_features.pkl") # Save features

        if os.path.exists(gallery_feature_path):
            print(f"    [CACHE] Loading gallery features: {gallery_feature_path}")
            features = load_features(gallery_feature_path)
        else:
            print(f"    [RUN] Extracting features for person ID {subject_id}...")
            features = extract_sil(gallery_silhouettes[gallery_video_path], feature_folder, subject_id)
            save_features(features, gallery_feature_path)
        
        gallery_features.update(features)  # <-- THIS is essential

    print("GALLERY FEATURES:", gallery_features)


    print("\nProcessing Probe Videos...")
    for probe_video_path in probe_video_paths:
        # video_name_base = probe_video_path.split("/")[-1].split(".")[0]
        # person_id = video_name_base.split("-")[0]

        path_parts = os.path.normpath(probe_video_path).split(os.sep)
        subject_id = path_parts[-3]
        view_id = path_parts[-2]
        seq_id = os.path.splitext(path_parts[-1])[0]
        video_name_base = f"{subject_id}-{view_id}-{seq_id}"

        video_save_folder = osp.join(video_output_dir, f"probe_{video_name_base}")
        os.makedirs(video_save_folder, exist_ok=True)

        print(f"  Processing: {probe_video_path}")
        
        track_path = osp.join(track_root, f"probe_{video_name_base}_track.pkl")
        probe_track_result = load_from_pickle(track_path)
        if probe_track_result is None:
            probe_track_result = track(probe_video_path, video_save_folder)
            save_to_pickle(probe_track_result, track_path)
        else:
            print(f"    [CACHE] Loaded tracking result: {track_path}")

        silhouette_folder = osp.join(silhouette_root, 'GaitSilhouette', f"probe_{video_name_base}")
        os.makedirs(silhouette_folder, exist_ok=True)
        probe_silhouette_path = osp.join(silhouette_folder, f"{video_name_base}_silhouette.pkl")

        if os.path.exists(probe_silhouette_path):
            print(f"    Loading existing silhouette: {probe_silhouette_path}")
            probe_silhouette = getsil(probe_video_path, silhouette_folder)
        else:
            print("    Segmenting silhouettes...")
            probe_silhouette = seg(probe_video_path, probe_track_result, silhouette_folder)
            save_features(probe_silhouette, probe_silhouette_path)

        feature_folder = osp.join(feature_root, 'GaitFeatures', f"probe_{video_name_base}")
        os.makedirs(feature_folder, exist_ok=True)
        
        probe_feature_path = osp.join(feature_folder, f"{video_name_base}_features.pkl")

        if os.path.exists(probe_feature_path):
            print(f"    [CACHE] Loading probe features: {probe_feature_path}")
            probe_features = load_features(probe_feature_path)
        else:
            print(f"    [RUN] Extracting features for person ID {subject_id}...")
            probe_features = extract_sil(probe_silhouette, feature_folder, subject_id)
            save_features(probe_features, probe_feature_path)


        print("PROBE FEATURE:", probe_features)

        print("    Performing Recognition...")
        # Flatten features before comparing
        # flat_probe_feat = []
        # for k, v in probe_features.items():
        #     if isinstance(v, list):
        #         flat_probe_feat.extend(v)

        # flat_gallery_feat = []
        # for k, v in gallery_features.items():
        #     if isinstance(v, list):
        #         flat_gallery_feat.extend(v)

        # recognition_results = compare(flat_probe_feat, flat_gallery_feat)

        recognition_results = compare(probe_features, gallery_features)

        print("probe_features type:", type(probe_features))
        print("gallery_features type:", type(gallery_features))

        # print("    Writing Results to Video...")
        # writeresult(recognition_results, probe_video_path, video_save_folder)

if __name__ == "__main__":
    main()