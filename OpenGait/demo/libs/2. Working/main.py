import os
import os.path as osp
import time
import sys
sys.path.append(os.path.abspath('.') + "/demo/libs/")
from track import *
from segment import *
from recognise import *

def main():
    output_dir = "./demo/output/OutputVideos_Multiple/"
    os.makedirs(output_dir, exist_ok=True)
    current_time = time.localtime()
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    main_video_save_folder = osp.join(output_dir, timestamp)
    os.makedirs(main_video_save_folder, exist_ok=True)

    save_root = './demo/output_multiple/'
    os.makedirs(save_root + 'GaitSilhouette', exist_ok=True)
    os.makedirs(save_root + 'GaitFeatures', exist_ok=True)

    gallery_video_paths = [
        "./demo/output/InputVideos/Set/1-1.mp4",
        # "./demo/output/InputVideos/Set/1-2.mp4",
        # "./demo/output/InputVideos/Set/1-3.mp4",
        # "./demo/output/InputVideos/Set/2-1.mp4",
        # "./demo/output/InputVideos/Set/2-2.mp4",
        # "./demo/output/InputVideos/Set/2-3.mp4",
        # Add more gallery video paths here
    ]

    probe_video_paths = [
        "./demo/output/InputVideos/Set/1-2.mp4",
        "./demo/output/InputVideos/Set/2-2.mp4",
        # "./demo/output/InputVideos/Set/1-3.mp4",
        # "./demo/output/InputVideos/Set/4-1.mp4",
        # Add more probe video paths here
    ]

    gallery_features = {}  # Dictionary to store gallery features (person_id: features)
    gallery_track_results = {} # Dictionary to store track results for galleries
    gallery_silhouettes = {} # Dictionary to store silhouette paths for galleries

    print("Processing Gallery Videos...")
    for gallery_video_path in gallery_video_paths:
        video_name_base = gallery_video_path.split("/")[-1].split(".")[0]
        person_id = video_name_base.split("-")[0]  # Extract person ID from video name
        video_save_folder = osp.join(main_video_save_folder, f"gallery_{video_name_base}")
        os.makedirs(video_save_folder, exist_ok=True)

        print(f"  Processing: {gallery_video_path}")
        track_result = track(gallery_video_path, video_save_folder)
        gallery_track_results[gallery_video_path] = track_result

        silhouette_folder = osp.join(save_root, 'GaitSilhouette', f"gallery_{video_name_base}")
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
            # Save the generated silhouettes for potential reuse (adapt to handle multiple people)
            # You might need a function to save the segmented outputs
            # For simplicity, we'll assume 'seg' saves them appropriately in the folder

        feature_folder = osp.join(save_root, 'GaitFeatures', f"gallery_{video_name_base}")
        os.makedirs(feature_folder, exist_ok=True)
        gallery_feature_path = osp.join(feature_folder, f"{video_name_base}_features.pkl") # Save features

        print("    Extracting features for person ID {person_id}...")
        features = extract_sil(gallery_silhouettes[gallery_video_path], feature_folder, person_id)
        # Assuming 'extract_sil' returns a dictionary of features per tracked ID
        gallery_features.update(features) # Merge features from different gallery videos
    
    print("GALLERY FEATURES:", gallery_features)


    print("\nProcessing Probe Videos...")
    for probe_video_path in probe_video_paths:
        video_name_base = probe_video_path.split("/")[-1].split(".")[0]
        person_id = video_name_base.split("-")[0]
        video_save_folder = osp.join(main_video_save_folder, f"probe_{video_name_base}")
        os.makedirs(video_save_folder, exist_ok=True)

        print(f"  Processing: {probe_video_path}")
        probe_track_result = track(probe_video_path, video_save_folder)

        silhouette_folder = osp.join(save_root, 'GaitSilhouette', f"probe_{video_name_base}")
        os.makedirs(silhouette_folder, exist_ok=True)
        probe_silhouette_path = osp.join(silhouette_folder, f"{video_name_base}_silhouette.pkl")

        if os.path.exists(probe_silhouette_path):
            print(f"    Loading existing silhouette: {probe_silhouette_path}")
            probe_silhouette = getsil(probe_video_path, silhouette_folder)
        else:
            print("    Segmenting silhouettes...")
            probe_silhouette = seg(probe_video_path, probe_track_result, silhouette_folder)

        feature_folder = osp.join(save_root, 'GaitFeatures', f"probe_{video_name_base}")
        os.makedirs(feature_folder, exist_ok=True)
        
        print(f"    Extracting features for person ID {person_id}...")
        probe_features = extract_sil(probe_silhouette, feature_folder, person_id)

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