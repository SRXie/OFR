# Generating test results for object algebra and attribute algebra
from clevr_obj_test.image_generation.obj_scene import Scene, Obj
from itertools import combinations
from math import log2, factorial
import os
import json
import itertools
import random

SCENE_SUMMARY = None

def create_path(test_root, main_scene_idx, sub_scene_idx=0, file_type='images'):
    assert file_type == "images" or file_type == "scenes" or file_type == "meta"
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
        if "/obj_test/" in meta_path:
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
            image_D_path = create_path(test_root, main_scene_idx, subset_idx_D)

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
                    image_C_path = create_path(test_root, main_scene_idx, subset_idx_C)

                    # hard negative
                    drop_idx = random.randint(0, len(subset_idx_list_A)-1)
                    subset_idx_E = scene.objs2img["-".join( str(idx) for idx in subset_idx_list_A[:drop_idx]+subset_idx_list_A[drop_idx+1:])]
                    image_E_path = create_path(test_root, main_scene_idx, subset_idx_E)

                    drop_idx = random.randint(0, len(subset_idx_list_D)-1)
                    subset_idx_F = scene.objs2img["-".join( str(idx) for idx in subset_idx_list_D[:drop_idx]+subset_idx_list_D[drop_idx+1:])]
                    image_F_path = create_path(test_root, main_scene_idx, subset_idx_F)

                    tuples.append((image_A_path, image_B_path, image_C_path, image_D_path, image_E_path, image_F_path))

            if not subset_idx_list_B in decomposed:
                tuples += obj_algebra_test(test_root, main_scene_idx, subset_idx_B, decomposed)
            if not subset_idx_list_D in decomposed:
                tuples += obj_algebra_test(test_root, main_scene_idx, subset_idx_D, decomposed)

    return tuples

def attr_algebra_test(test_root, main_scene_idx=0, sub_scene_idx=0):
    # there are four attributes of interest, we edit at most 3

    # We first create scene from json
    scene = create_scene_from_path(test_root, main_scene_idx, sub_scene_idx)
    tuples = [] # path tuples

    # create algebra tuples A-B+C=D: Scene - Edit_complementary + Edit_all = Edit_selected
    image_A_path = create_path(test_root, main_scene_idx)
    for num_edit in range(1, len(scene.get_obj_attrs())):
        # iterate through attribute combinations of the scene
        for attr_subset in combinations(scene.get_obj_attrs(), num_edit):
            attr_complementary = scene.get_obj_attrs().difference(attr_subset)

            image_D_idxs = scene.objects[0].edit_attrs(list(attr_subset), return_imgs=True) # convert set to list to enforce order for matching itertools.prod(D, B) and C
            image_B_idxs = scene.objects[0].edit_attrs(list(attr_complementary), return_imgs=True)
            image_C_idxs = scene.objects[0].edit_attrs(list(attr_subset)+list(attr_complementary), return_imgs=True)
            # Hard negatives
            image_E_idxs = image_C_idxs.copy()
            random.shuffle(image_E_idxs)
            i = 0
            for d_idx in image_D_idxs:
                for b_idx  in image_B_idxs:
                    c_idx = image_C_idxs[i]
                    e_idx = image_E_idxs[i]
                    image_B_path = create_path(test_root, main_scene_idx, b_idx)
                    image_C_path = create_path(test_root, main_scene_idx, c_idx)
                    image_D_path = create_path(test_root, main_scene_idx, d_idx)
                    image_E_path = create_path(test_root, main_scene_idx, e_idx)
                    tuples.append((image_A_path, image_B_path, image_C_path, image_D_path, image_E_path))
                    i+=1
            assert i == len(image_C_idxs)
    return tuples
