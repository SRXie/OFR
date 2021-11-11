import pandas as pd;
from itertools import combinations, product

seed_list = [0,5,6,7,11,13]#,17,18,19,20]
data_list = ["iid", "shape_0", "size_0", "color_0", "material_0","appr_corr_0", "material_0shape_0", "shape_0color_0", "size_0material_0", "material_0shape_0color_0", "size_0material_0color_0", "size_0material_0shape_0", "size_0shape_0color_0", "intr_corr_0"]
df = pd.DataFrame(columns=["iid", "shape_0", "size_0", "color_0", "material_0","appr_corr_0", "material_0shape_0", "shape_0color_0", "size_0material_0", "material_0shape_0color_0", "size_0material_0color_0", "size_0material_0shape_0", "size_0shape_0color_0", "intr_corr_0", "seed"])

# data_list = [ ["iid"], ["appr_corr_0", "appr_corr_0.5_mat_0"], ["intr_corr_0", "intr_corr_0.5"], ["pos_corr_1.5", "pos_mutex_1.5"] ]

# one dataset
r = 0
for ds in data_list:
    for seed in seed_list:
        d1 = {ds: 1.0, "seed": seed}
        df.at[str(r), :] = d1
        r +=1

# two datasets
for ds in data_list[1:]:
    for seed in seed_list:
        # d1 = {"iid": 0.9, ds:0.1, "seed": seed}
        d1 = {"iid": 0.75, ds:0.25, "seed": seed}
        d2 = {"iid": 0.5, ds:0.5, "seed": seed}
        d3 = {"iid": 0.25, ds:0.75, "seed": seed}
        # d5 = {"iid": 0.1, ds:0.9, "seed": seed}
        df.at[str(r), :] = d1
        df.at[str(r+1), :] = d2
        df.at[str(r+2), :] = d3
        #df.at[str(r+3), :] = d4
        #df.at[str(r+4), :] = d5
        r +=3

# # three datasets
# for comb in combinations(data_list, 3):
#     for prod in product(comb[0], comb[1], comb[2]):
#         for seed in range(2):
#             d1 = {prod[0]: 0.1, prod[1]:0.45, prod[2]:0.45, "seed": seed}
#             d2 = {prod[0]: 0.45, prod[1]:0.1, prod[2]:0.45, "seed": seed}
#             d3 = {prod[0]: 0.45, prod[1]:0.45, prod[2]:0.1, "seed": seed}
#             d4 = {prod[0]: 0.33, prod[1]:0.34, prod[2]:0.33, "seed": seed}
#             df.at[str(r), :] = d1
#             df.at[str(r+1), :] = d2
#             df.at[str(r+2), :] = d3
#             df.at[str(r+3), :] = d4
#             r += 4

# # four datasets
# for comb in combinations(data_list, 4):
#     for prod in product(comb[0], comb[1], comb[2], comb[3]):
#         for seed in range(2):
#             d1 = {prod[0]: 0.1, prod[1]:0.3, prod[2]:0.3, prod[3]:0.3, "seed": seed}
#             d2 = {prod[0]: 0.3, prod[1]:0.1, prod[2]:0.3, prod[3]:0.3, "seed": seed}
#             d3 = {prod[0]: 0.3, prod[1]:0.3, prod[2]:0.1, prod[3]:0.3, "seed": seed}
#             d4 = {prod[0]: 0.3, prod[1]:0.3, prod[2]:0.3, prod[3]:0.1, "seed": seed}
#             d5 = {prod[0]: 0.25, prod[1]:0.25, prod[2]:0.25, prod[3]:0.25, "seed": seed}
#             df.at[str(r), :] = d1
#             df.at[str(r+1), :] = d2
#             df.at[str(r+2), :] = d3
#             df.at[str(r+3), :] = d4
#             df.at[str(r+4), :] = d5
#             r += 5

df.to_csv("/checkpoint/siruixie/data_mix.csv")
