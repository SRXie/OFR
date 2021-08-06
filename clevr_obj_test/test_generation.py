# Generating test results for object algebra and attribute algebra
from clevr_obj_test.image_generation.obj_scene import Scene, Obj
from itertools import combinations
from math import log2, factorial
import os
import json

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
    image_B_path = create_path(test_root, -1) # the -1th image should be the background image
    image_D_path = create_path(test_root, main_scene_idx, sub_scene_idx)
    for num_decomp in range(1, len(scene.objs_idx)):
        # generate all object algebra test for the given scene
        # First decompose it into subscenes:
        ACs_pairs = scene.decompose(num_decomp)
        decomposed.append(sub_scene_idx)
        # create algebra tuples A-B+C=D: Subscene_1 - Background + Subscene_2 = Scene
        for AC_pair in ACs_pairs:
            image_A_path = create_path(test_root, main_scene_idx, AC_pair[0])
            image_C_path = create_path(test_root, main_scene_idx, AC_pair[1])
            tuples.append((image_A_path, image_B_path, image_C_path, image_D_path))
            if not AC_pair[0] in decomposed:
                tuples += obj_algebra_test(test_root, main_scene_idx, AC_pair[0], decomposed)
            if not AC_pair[1] in decomposed:
                tuples += obj_algebra_test(test_root, main_scene_idx, AC_pair[1], decomposed)

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

            image_D_idxs = scene.objects[0].edit_attrs(list(attr_subset), return_imgs=True) # converst set to list to enforce order for matching itertools.prod(D, B) and C
            image_B_idxs = scene.objects[0].edit_attrs(list(attr_complementary), return_imgs=True)
            image_C_idxs = scene.objects[0].edit_attrs(list(attr_subset)+list(attr_complementary), return_imgs=True)
            i = 0
            for d_idx in image_D_idxs:
                for b_idx  in image_B_idxs:
                    c_idx = image_C_idxs[i]
                    image_B_path = create_path(test_root, main_scene_idx, b_idx)
                    image_C_path = create_path(test_root, main_scene_idx, c_idx)
                    image_D_path = create_path(test_root, main_scene_idx, d_idx)
                    tuples.append((image_A_path, image_B_path, image_C_path, image_D_path))
                    i+=1
            assert i == len(image_C_idxs)
    return tuples
