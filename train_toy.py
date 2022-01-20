from csv import reader
import os
import json
import numpy as np
import torch
from utils import compute_loss, compute_cosine_loss
from utils import compute_shuffle_loss, compute_shuffle_cosine_loss
from utils import summarize_losses

with open(os.path.join("/checkpoint/siruixie/clevr_obj_test/output/obj_test_occ_prep/3/", "obj_test", "CLEVR_test_cases.csv"), "r") as f:
    csv_reader = reader(f)
    obj_algebra_test_cases = list(csv_reader)

A, B, C, D = [], [], [], []
ABCD = [A, B, C, D]

color_map = {
    "gray": -0.87,
    "red": 0.73,
    "blue": -0.42,
    "green": 0.29,
    "brown": 0.33,
    "purple": -0.15,
    "cyan": -0.66,
    "yellow": 0.52
    }

shape_map = {
    "cube": -0.3,
    "sphere": -0.6,
    "cylinder": -0.9
}

for test_case in obj_algebra_test_cases:

    test_case = test_case[:4] # get A, B, C, D
    test_case = [tstr.replace("images", "scenes") for tstr in test_case]
    test_case = [tstr.replace("bgs", "scenes") for tstr in test_case]
    test_case = [tstr.replace("png", "json") for tstr in test_case]

    for index, scene_path in enumerate(test_case):
        with open(scene_path) as f:
            scene = json.load(f)
            obj_vector = -np.ones((9, 5))
            for obj_dict in scene["objects"]:
                x = obj_dict["threed_coords"][0]
                y = obj_dict["threed_coords"][1]
                size = obj_dict["scale"]
                shape = shape_map[obj_dict["shape"]]
                color = color_map[obj_dict["color"]]
                obj_vector[obj_dict["index"]] = np.array([x, y, size, shape, color])

            ABCD[index].append(obj_vector)

A = torch.tensor((A)).cuda().view(-1, 45)
B = torch.tensor((B)).cuda().view(-1, 45)
C = torch.tensor((C)).cuda().view(-1, 45)
D = torch.tensor((D)).cuda().view(-1, 45)

print("Sampling a random full-rank matrix...")

while True:
    proj = 1000*(torch.rand(45, 45)-0.5)
    if torch.matrix_rank(proj) == 45:
        proj = proj.cuda()
        break
    else:
        print(torch.matrix_rank(proj))

while True:
    proj2 = 1000*(torch.rand(45, 45)-0.5)
    if torch.matrix_rank(proj2) == 45:
        proj2 = proj2.cuda()
        break
    else:
        print(torch.matrix_rank(proj2))

elu = torch.nn.ELU()

obj_losses, obj_losses_en_D = [], []
obj_cos_losses, obj_cos_losses_en_D = [], []
obj_acos_losses, obj_acos_losses_en_D = [], []

for i in range(1, len(obj_algebra_test_cases)//64):
    z_A = torch.tanh(torch.einsum("bk, kk -> bk", A[(i-1)*64:i*64], proj))
    z_B = torch.tanh(torch.einsum("bk, kk -> bk", B[(i-1)*64:i*64], proj))
    z_C = torch.tanh(torch.einsum("bk, kk -> bk", C[(i-1)*64:i*64], proj))
    z_D = torch.tanh(torch.einsum("bk, kk -> bk", D[(i-1)*64:i*64], proj))
    
    z_A = (torch.einsum("bk, kk -> bk", z_A, proj2))
    z_B = (torch.einsum("bk, kk -> bk", z_B, proj2))
    z_C = (torch.einsum("bk, kk -> bk", z_C, proj2))
    z_D = (torch.einsum("bk, kk -> bk", z_D, proj2))
    print(A[(i-1)*64])
    print(B[(i-1)*64])
    print(C[(i-1)*64])
    print(D[(i-1)*64])

    print(z_A[0]-z_B[0])
    print(z_D[0]-z_C[0])
    print("-----------------")
    cat_zs = torch.cat([z_A, z_B, z_C, z_D], 0)

    compute_loss(cat_zs, obj_losses)
    compute_cosine_loss(cat_zs, obj_cos_losses, obj_acos_losses)

    compute_shuffle_loss(cat_zs, obj_losses_en_D)
    compute_shuffle_cosine_loss(cat_zs, obj_cos_losses_en_D, obj_acos_losses_en_D)

std_obj_l2_ratio, avg_obj_l2_ratio, avg_obj_l2, avg_obj_l2_baseline = summarize_losses(obj_losses, obj_losses_en_D)
std_obj_cos_ratio, avg_obj_cos_ratio, avg_obj_cos, avg_obj_cos_baseline = summarize_losses(obj_cos_losses, obj_cos_losses_en_D)
std_obj_acos_ratio, avg_obj_acos_ratio, avg_obj_acos, avg_obj_acos_baseline = summarize_losses(obj_acos_losses, obj_acos_losses_en_D)

print("avg_obj_l2_ratio:", avg_obj_l2_ratio)
print("avg_obj_cos_ratio:", avg_obj_cos_ratio)
print("avg_obj_acos_ratio:", avg_obj_acos_ratio)
