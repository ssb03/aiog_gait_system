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
        ipts = gaitmodel.inputs_pretreament(inputs)
        id = person_id  # ← dynamic ID passed from main.py
        if id not in feats:
            feats[id] = []
        type = inputs[2][0] 
        view = inputs[3][0]
        embs_pkl_path = "{}/{}/{}/{}".format(embs_save_path, id, type, view)
        if not os.path.exists(embs_pkl_path):
            os.makedirs(embs_pkl_path)
        embs_pkl_name = "{}/{}.pkl".format(embs_pkl_path, inputs[3][0])
        retval, embs = gaitmodel.forward(ipts)
        pkl = open(embs_pkl_name, 'wb')
        pickle.dump(embs, pkl)
        feat = {}
        feat[type] = {}
        feat[type][view] = embs
        feats[id].append(feat)        
    return feats    

def gaitfeat_compare(probe_feat:dict, gallery_feat:dict):
    """Compares the feature between probe and gallery

    Args:
        probe_feat (dict): Dictionary of probe's features
        gallery_feat (dict): Dictionary of gallery's features
    Returns:
        pg_dicts (dict): The id of probe corresponds to the id of gallery
    """
    # item = list(probe_feat.keys())
    # probe = item[0]
    # pg_dict = {}
    # pg_dicts = {}
    # for inputs in probe_feat[probe]:
    #     number = list(inputs.keys())[0]
    #     probeid = probe + "-" + number
    #     galleryid, idsdict = gc.comparefeat(inputs[number]['undefined'], gallery_feat, probeid, 100)
    #     pg_dict[probeid] = galleryid
    #     pg_dicts[probeid] = idsdict
    # # print("=================== pg_dicts ===================")
    # # print(pg_dicts)
    # return pg_dict

    pg_dict = {}
    pg_dicts = {}

    for probe_id, probe_data_list in probe_feat.items():  # Iterate through probe IDs
        for probe_data in probe_data_list:  # Iterate through list of features for each probe
            for probe_type, probe_views in probe_data.items():  # Iterate through types
                for probe_view, probe_embeddings in probe_views.items():  # Iterate through views
                    probeid = f"{probe_id}-{probe_type}"  # Construct a unique ID
                    galleryid, idsdict = gc.comparefeat(probe_embeddings, gallery_feat, probeid, 0.02)
                    pg_dict[probeid] = galleryid
                    pg_dicts[probeid] = idsdict

    return pg_dict

def extract_sil(sil, save_path):
    """Gets the features.

    Args:
        sils (list): List of Tuple (seqs, labs, typs, vies, seqL)
        save_path (Path): Output path.
    Returns:
        video_feats (dict): Dictionary of features from the video
    """
    logger.info("begin extracting")
    video_feat = gait_sil(sil, save_path)
    logger.info("extract Done")
    return video_feat


def compare(probe_feat, gallery_feat):
    """Recognizes the features between probe and gallery

    Args:
        probe_feat (dict): Dictionary of probe's features
        gallery_feat (dict): Dictionary of gallery's features
    Returns:
        pgdict (dict): The id of probe corresponds to the id of gallery
    """
    logger.info("Begin recognizing")

    pgdict = gaitfeat_compare(probe_feat, gallery_feat)

    # Ensure all values are safe for parsing in writeresult()
    for k in pgdict:
        if pgdict[k] is None or not isinstance(pgdict[k], str) or "-" not in pgdict[k]:
            pgdict[k] = "no_match"

    logger.info("Recognition Done")
    print("================= probe - gallery ===================")
    print(pgdict)
    return pgdict
