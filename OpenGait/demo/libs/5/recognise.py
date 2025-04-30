import os
import os.path as osp
import pickle
import sys
# import shutil

root = os.path.dirname(os.path.dirname(os.path.dirname( os.path.abspath(__file__) )))
sys.path.append(root)
from opengait.utils import config_loader
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname( os.path.abspath(__file__)))) + "/modeling/")
from loguru import logger
import model.baselineDemo as baselineDemo
import gait_compare as gc

recognise_cfgs = {  
    "gaitmodel":{
        "model_type": "BaselineDemo",
        # "cfg_path": "./configs/baseline/baseline_GREW.yaml",
        "cfg_path": "./configs/gaitbase/gaitbase_da_gait3d.yaml",
    },
}


def loadModel(model_type, cfg_path):
    Model = getattr(baselineDemo, model_type)
    cfgs = config_loader(cfg_path)
    model = Model(cfgs, training=False)
    return model

def gait_sil(sils, embs_save_path, person_id = '001'):
    """Gets the features.

    Args:
        sils (list): List of Tuple (seqs, labs, typs, vies, seqL)
        embs_save_path (Path): Output path.
    Returns:
        feats (dict): Dictionary of features
    """
    gaitmodel = loadModel(**recognise_cfgs["gaitmodel"])
    gaitmodel.requires_grad_(False)
    gaitmodel.eval()
    feats = {}
    
    for inputs in sils:
        print(f"[DEBUG] inputs = {inputs}")
        id = person_id  # Provided externally (main.py)       
        typ = inputs[2][0]
        view = inputs[3][0]

        embs_dir = os.path.join(embs_save_path, id, typ, view)
        os.makedirs(embs_dir, exist_ok=True)
        embs_pkl_name = os.path.join(embs_dir, f"{view}.pkl")

        # ✅ Reuse precomputed embeddings if they exist
        if os.path.exists(embs_pkl_name):
            with open(embs_pkl_name, 'rb') as f:
                embs = pickle.load(f)
            logger.info(f"Reusing saved embedding: {embs_pkl_name}")
            print(f"[DEBUG] Reused embedding from {embs_pkl_name}")  # Debug: Check reuse

        else:
            ipts = gaitmodel.inputs_pretreament(inputs)
            print(f"[DEBUG] Preprocessed inputs: {ipts}")  # Debug: Check preprocessed inputs
            _, embs = gaitmodel.forward(ipts)
            print(f"[DEBUG] Embedding shape: {embs.shape if hasattr(embs, 'shape') else len(embs)}")  # Debug: Check shape of embeddings
            with open(embs_pkl_name, 'wb') as f:
                pickle.dump(embs, f)
            logger.info(f"Saving embedding: {embs_pkl_name}")

        # Organize features for return
        if id not in feats:
            feats[id] = {}
        if typ not in feats[id]:
            feats[id][typ] = {}
            feats[id][typ][view] = embs
    return feats   

# def gaitfeat_compare(probe_feat:dict, gallery_feat:dict):
#     """Compares the feature between probe and gallery

#     Args:
#         probe_feat (dict): Dictionary of probe's features
#         gallery_feat (dict): Dictionary of gallery's features
#     Returns:
#         pg_dicts (dict): The id of probe corresponds to the id of gallery
#     """
#     # item = list(probe_feat.keys())
#     # probe = item[0]
#     # pg_dict = {}
#     # pg_dicts = {}
#     # for inputs in probe_feat[probe]:
#     #     number = list(inputs.keys())[0]
#     #     probeid = probe + "-" + number
#     #     galleryid, idsdict = gc.comparefeat(inputs[number]['undefined'], gallery_feat, probeid, 100)
#     #     pg_dict[probeid] = galleryid
#     #     pg_dicts[probeid] = idsdict
#     # # print("=================== pg_dicts ===================")
#     # # print(pg_dicts)
#     # return pg_dict

#     pg_dict = {}
#     pg_dicts = {}

#     # for probe_id, probe_data_list in probe_feat.items():  # Iterate through probe IDs
#     #     for probe_data in probe_data_list:  # Iterate through list of features for each probe
#     #         for probe_type, probe_views in probe_data.items():  # Iterate through types
#     #             for probe_view, probe_embeddings in probe_views.items():  # Iterate through views
#     #                 probeid = f"{probe_id}-{probe_type}"  # Construct a unique ID
#     #                 galleryid, idsdict = gc.comparefeat(probe_embeddings, gallery_feat, probeid, 0.02)
#     #                 pg_dict[probeid] = galleryid
#     #                 pg_dicts[probeid] = idsdict

    
#         # for probe_type, probe_views in probe_data.items():  # Iterate through types
#         #     for probe_view, probe_embeddings in probe_views.items():  # Iterate through views
#         #         probeid = f"{probe_id}-{probe_type}"  # Construct a unique ID
#         #         galleryid, idsdict = gc.comparefeat(probe_embeddings, gallery_feat, probeid, 0.02)
#         #         pg_dict[probeid] = galleryid
#         #         pg_dicts[probeid] = idsdict
    
#     for probe_id, probe_data in probe_feat.items():  # Iterate through probe IDs
#         for probeid, probes in probe_feat.items():
#             for seqid, probe_views in probes.items():
#                 for view, probe_emb in probe_views.items():
#                     # Handle comparison with each subject in gallery
#                     for galid, gal_sequences in gallery_feat.items():
#                         for typ, subject in gal_sequences.items():
#                             if isinstance(subject, dict):
#                                 for gview, gallery_emb in subject.items():
#                                     # do your cosine similarity etc.
#                                     pass
#                             else:
#                                 print(f"Expected dict at subject[{typ}], got {type(subject)}: {subject}")
#     return pg_dict

def gaitfeat_compare(probe_feat: dict, gallery_feat: dict, sim_threshold: float = 0.02, top_k: int = 3):
    """Compares probe features with gallery features using cosine similarity.

    Args:
        probe_feat (dict): Dictionary of probe's features
        gallery_feat (dict): Dictionary of gallery's features
        sim_threshold (float): Similarity threshold for matching

    Returns:
        pg_dict (dict): Mapping of probe IDs to best matching gallery IDs
        pg_dicts (dict): Raw similarity dictionary per probe
    """
    pg_dict = {}
    pg_dicts = {}

    for probe_id, probe_data in probe_feat.items():
        for probe_type, probe_views in probe_data.items():
            for view, probe_emb in probe_views.items():
                probe_unique_id = f"{probe_id}-{probe_type}"
                print(f"[DEBUG] Comparing {probe_unique_id} (view={view})")  # Debug: Check which probe is being compared
                # Compare this embedding with the entire gallery
                try:
                    galleryid, sorted_matches = gc.comparefeat(probe_emb, gallery_feat, probe_unique_id, sim_threshold)
                    pg_dict[probe_unique_id] = galleryid
                    # ✅ Save top-k results only
                    pg_dicts[probe_unique_id] = sorted_matches[:top_k]
                    logger.info(f"Top-{top_k} matches for {probe_unique_id}:")
                    print(f"[DEBUG] Top-{top_k} matches for {probe_unique_id}: {sorted_matches[:top_k]}")  # Debug: Check top-k matches
                    for rank, (gid, dist) in enumerate(sorted_matches[:top_k], 1):
                        logger.info(f"{rank}. {gid} (distance={dist:.4f})")
                except Exception as e:
                    logger.error(f"Error comparing probe {probe_unique_id}: {e}")
                    pg_dict[probe_unique_id] = "no_match"
                    pg_dicts[probe_unique_id] = []

    return pg_dict, pg_dicts

def extract_sil(sil, save_path, person_id='001'):
    """Gets the features.

    Args:
        sil (list): List of Tuple (seqs, labs, typs, vies, seqL)
        save_path (Path): Output path.
    Returns:
        video_feats (dict): Dictionary of features from the video
    """
    logger.info("Begin extracting...")

    embs_path = os.path.join(save_path, person_id)
    video_feat = {person_id: {}}

    for inputs in sil:
        print(f"[DEBUG] Full input tuple: {inputs}")
        print(f"[DEBUG] inputs[3] (view): {inputs[3]}")
        typ = inputs[2][0]
        if len(inputs[3]) == 0:
            logger.warning(f"Empty 'vies' for {inputs}, setting view to 'undefined'")
            view = "undefined"
        else:
            view = inputs[3][0]
        embs_pkl_path = os.path.join(embs_path, typ, view, f"{view}.pkl")

        if os.path.exists(embs_pkl_path):
            logger.info(f"[CACHE] Reusing embedding: {embs_pkl_path}")
            print(f"[DEBUG] Reusing cached embedding from {embs_pkl_path}")  # Debug: Check cached path
            with open(embs_pkl_path, 'rb') as f:
                embs = pickle.load(f)
        else:
            logger.info(f"[RUN] Extracting embedding for: ID={person_id}, type={typ}, view={view}")
            gaitmodel = loadModel(**recognise_cfgs["gaitmodel"])
            gaitmodel.requires_grad_(False)
            gaitmodel.eval()
            ipts = gaitmodel.inputs_pretreament(inputs)
            # print(f"[DEBUG] Preprocessed inputs: {ipts}")  # Debug: Check preprocessed inputs
            _, embs = gaitmodel.forward(ipts)
            # print(f"[DEBUG] Embedding shape: {embs.shape if hasattr(embs, 'shape') else len(embs)}")  # Debug: Check shape of embeddings

            os.makedirs(os.path.dirname(embs_pkl_path), exist_ok=True)
            with open(embs_pkl_path, 'wb') as f:
                pickle.dump(embs, f)

        # Store into proper structure
        if typ not in video_feat[person_id]:
            video_feat[person_id][typ] = {}
        video_feat[person_id][typ][view] = embs

    logger.info("Extract Done.")
    return video_feat



def compare(probe_feat, gallery_feat, top_k=3):
    """Recognizes the features between probe and gallery

    Args:
        probe_feat (dict): Dictionary of probe's features
        gallery_feat (dict): Dictionary of gallery's features
    Returns:
        pgdict (dict): The id of probe corresponds to the id of gallery
    """
    logger.info("Begin recognizing")

    pgdict, topk_dicts = gaitfeat_compare(probe_feat, gallery_feat, top_k=top_k)

    for k in pgdict:
        if pgdict[k] is None or not isinstance(pgdict[k], str) or "-" not in pgdict[k]:
            pgdict[k] = "no_match"

    logger.info("Recognition Done")
    print("================= probe - gallery ===================")
    for probe_id in pgdict:
        print(f"{probe_id} → {pgdict[probe_id]}")
        print("Top Matches:")
        for rank, (gid, dist) in enumerate(topk_dicts.get(probe_id, []), 1):
            print(f"  {rank}. {gid} (distance={dist:.4f})")
    return pgdict
