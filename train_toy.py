from csv import reader
import json
from utils import compute_loss, compute_cosine_loss
from utils import compute_shuffle_loss, compute_shuffle_cosine_loss
from utils import summarize_losses

with open(os.path.join("/checkpoint/siruixie/clevr_obj_test/output/obj_test_occ_prep/3/", "obj_test", "CLEVR_test_cases.csv"), "r") as f:
    csv_reader = reader(f)
    obj_algebra_test_cases = list(csv_reader)

A, B, C, D = [], [], [], []
ABCD = [A, B, C, D]

for test_case in obj_algebra_test_cases:

    test_case = test_case[:4] # get A, B, C, D
    test_case = [tstr.replace("images", "scenes") for tstr in test_case]
    test_case = [tstr.replace("bgs", "scenes") for tstr in test_case]
    test_case = [tstr.replace("png", "json") for tstr in test_case]

    for index, scene_path in enumerate(test_case):
        with open(scene_path) as f:
            scene = json.load(f)
            obj_vector = np.zeros(9, 5)
            for obj_dict in scene["objects"]:
                x = obj_dict["threed_coords"][0]
                y = obj_dict["threed_coords"][1]
                size = obj_dict["scale"]
                shape = shape_map[obj_dict["shape"]]
                color = color_map[obj_dict["color"]]
                obj_vector[obj_dict["index"]] = np.array([x, y, size, shape, color])

            ABCD[index].append(obj_vector)

A = torch.tensor(A).cuda()
B = torch.tensor(B).cuda()
C = torch.tensor(C).cuda()
D = torch.tensor(D).cuda()

print("Sampling a random full-rank matrix...")

while True:
    proj = torch.rand(45, 45)
    if torch.matrix_rank(proj) == 45:
        break
    else:
        print torch.matrix_rank(proj)

obj_losses, obj_losses_en_D = [], []
obj_cos_losses, obj_cos_losses_en_D = [], []
obj_acos_losses, obj_acos_losses_en_D = [], []

for i in range(len(obj_algebra_test_cases)//64):
    z_A = torch.einsum("bk, kk -> bk", A, proj)
    z_B = torch.einsum("bk, kk -> bk", B, proj)
    z_C = torch.einsum("bk, kk -> bk", C, proj)
    z_D = torch.einsum("bk, kk -> bk", D, proj)

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
