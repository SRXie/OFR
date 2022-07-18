# Generating test results for object algebra and attribute algebra
from clevr_obj_test.image_generation.obj_scene import Scene, Obj
from itertools import combinations
from math import log2, factorial
import os
import json
import itertools
import random
from PIL import Image
import numpy as np
from copy import deepcopy

SCENE_SUMMARY = None

def create_path(test_root, main_scene_idx, sub_scene_idx=0, file_type='images'):
    assert file_type == "images" or file_type == "scenes" or file_type == "meta" or file_type == "masks" or file_type == "bgs"
    global SCENE_SUMMARY
    if not SCENE_SUMMARY:
        with open(os.path.join(test_root, "CLEVR_scenes.json")) as f:
            SCENE_SUMMARY = json.load(f)
    if main_scene_idx == -1:
        main_scene_idx = 0
        path = os.path.join(test_root, file_type, SCENE_SUMMARY["scenes"][main_scene_idx]["image_filename"])
        path = path[:-5]+'0'+path[-4:]
    else:
        path = os.path.join(test_root, file_type, SCENE_SUMMARY["scenes"][main_scene_idx]["image_filename"])
    if not sub_scene_idx == 0 and not file_type == "meta":
        path = path[:-4]+'_%04d' % sub_scene_idx+path[-4:]
    if file_type == "scenes" or file_type == "meta":
        path = path[:-4]+'.json'
    return path

def create_scene_from_path(test_root, main_scene_idx, sub_scene_idx=0):
    scene_path = create_path(test_root, main_scene_idx, sub_scene_idx, file_type="scenes")
    meta_path = create_path(test_root, main_scene_idx, sub_scene_idx, file_type="meta")
    # TODO: open json from path and create Scene with it
    with open(scene_path) as f1:
        scene_dict = json.load(f1)

    with open(meta_path) as f2:
        if "/obj_test_final/" in meta_path:
            objs2img = json.load(f2)
            attrs2img = None
        elif "/attr_test/" in meta_path:
            objs2img = None
            attrs2img = json.load(f2)
        else:
            raise NotImplementedError

    scene = Scene(split=scene_dict["split"], image_index=scene_dict["image_index"],
        image_filename=scene_dict["image_filename"], objects=[],
        directions=scene_dict["directions"],objs_idx=scene_dict["objs_idx"],
        objs2img=objs2img,bg_image_index=scene_dict["bg_image_index"])
    for od in scene_dict["objects"]:
        obj = Obj(index=od["index"], shape=od["shape"], size=od["size"],
        scale=od["scale"], material=od["material"], threed_coords=od["threed_coords"],
        rotation=od["rotation"], pixel_coords=od["pixel_coords"], color=od["color"],
        attrs2img=attrs2img,)
        scene.objects.append(obj)
    return scene

def obj_algebra_test(test_root, main_scene_idx=0, sub_scene_idx=0, decomposed=None):
    # We first create scene from json
    scene = create_scene_from_path(test_root, main_scene_idx, sub_scene_idx)

    num_objs = len(scene.objs_idx)
    if num_objs==1:
        return []

    if not decomposed:
        decomposed = []
    tuples = [] # path tuples
    decomposed.append(sub_scene_idx)
    for num_decomp in range(3, min(len(scene.objs_idx)-2, 6)):
        # generate all object algebra test for the given scene
        # First decompose it into part_23 (num_obj>=2) and part_1:
        for part_1 in itertools.combinations(set(scene.objs_idx), num_decomp):
            part_23 = set(scene.objs_idx).difference(set(part_1))

            subset_idx_list_B = sorted(list(part_1))
            subset_idx_list_D = sorted(list(part_23))

            subset_idx_B = scene.objs2img["-".join( str(idx) for idx in subset_idx_list_B)]
            subset_idx_D = scene.objs2img["-".join( str(idx) for idx in subset_idx_list_D)]

            image_B_path = create_path(test_root, main_scene_idx, subset_idx_B)
            image_D_path = create_path(test_root, main_scene_idx, subset_idx_D, file_type="bgs")

            mask_B_path = create_path(test_root, main_scene_idx, subset_idx_B, file_type="masks")
            mask_D_path = create_path(test_root, main_scene_idx, subset_idx_D, file_type="masks")

            # Then decompose part_23 into part_2 and part_3:
            for num_decomp_2 in range(1,min(len(part_23)-2,7-len(part_1))):
                for part_2 in itertools.combinations(set(part_23), num_decomp_2):
                    part_3 = set(part_23).difference(set(part_2))

                    # Recombine them to have A=part_12, B=part_1, C=part_3, D=part_23
                    subset_idx_list_A = sorted(list(set(part_1).union(set(part_2))))
                    subset_idx_list_C = sorted(list(part_3))

                    subset_idx_A = scene.objs2img["-".join( str(idx) for idx in subset_idx_list_A)]
                    subset_idx_C = scene.objs2img["-".join( str(idx) for idx in subset_idx_list_C)]

                    image_A_path = create_path(test_root, main_scene_idx, subset_idx_A)
                    image_C_path = create_path(test_root, main_scene_idx, subset_idx_C, file_type="bgs")

                    mask_A_path = create_path(test_root, main_scene_idx, subset_idx_A, file_type="masks")
                    mask_C_path = create_path(test_root, main_scene_idx, subset_idx_C, file_type="masks")

                    try:
                        mask_A = np.array(Image.open(mask_A_path))[:,:,0]
                        mask_A = np.where(mask_A==64, 0.0, mask_A)
                        mask_A = np.where(mask_A==255, 1.0, mask_A)
                        mask_B = np.array(Image.open(mask_B_path))[:,:,0]
                        mask_B = np.where(mask_B==64, 0.0, mask_B)
                        mask_B = np.where(mask_B==255, 1.0, mask_B)
                        mask_C = np.array(Image.open(mask_C_path))[:,:,0]
                        mask_C = np.where(mask_C==64, 0.0, mask_C)
                        mask_C = np.where(mask_C==255, 1.0, mask_C)
                        mask_D = np.array(Image.open(mask_D_path))[:,:,0]
                        mask_D = np.where(mask_D==64, 0.0, mask_D)
                        mask_D = np.where(mask_D==255, 1.0, mask_D)

                        if np.abs(mask_A-mask_B+mask_C-mask_D).sum() < 400.0: #> 1400.0: # 1200 is the minimum number of visible pixels
                            # hard negative
                            drop_idx_d = random.randint(0, len(subset_idx_list_D)-2)
                            subset_idx_list_E = deepcopy(subset_idx_list_D)
                            subset_idx_list_E =  subset_idx_list_E[:drop_idx_d]+list(subset_idx_list_E[drop_idx_d+1:])
                            subset_idx_E = scene.objs2img["-".join( str(idx) for idx in subset_idx_list_E)]
                            image_E_path = create_path(test_root, main_scene_idx, subset_idx_E, file_type="bgs")

                            replace_idx_b = random.randint(0, len(subset_idx_list_B)-1)
                            replace_idx_d = random.randint(0, len(subset_idx_list_D)-1)
                            subset_idx_list_F = deepcopy(subset_idx_list_D)
                            subset_idx_list_F[replace_idx_d] =  subset_idx_list_B[replace_idx_b]
                            subset_idx_list_F = sorted(subset_idx_list_F)
                            subset_idx_F = scene.objs2img["-".join( str(idx) for idx in subset_idx_list_F)]
                            image_F_path = create_path(test_root, main_scene_idx, subset_idx_F, file_type="bgs")

                            image_G_path = image_D_path.replace("/bgs/", "/color/")
                            image_H_path = image_D_path.replace("/bgs/", "/material/")
                            image_I_path = image_D_path.replace("/bgs/", "/shape/")
                            image_J_path = image_D_path.replace("/bgs/", "/size/")

                            tuples.append((image_A_path, image_B_path, image_C_path, image_D_path, image_E_path, image_F_path, image_G_path, image_H_path, image_I_path, image_J_path))
                    except Exception as e:
                        print(e)

            if not subset_idx_list_B in decomposed:
                tuples += obj_algebra_test(test_root, main_scene_idx, subset_idx_B, decomposed)
            if not subset_idx_list_D in decomposed:
                tuples += obj_algebra_test(test_root, main_scene_idx, subset_idx_D, decomposed)

    return tuples

