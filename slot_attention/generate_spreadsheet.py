import pandas as pd;
from itertools import combinations, product


df = pd.DataFrame(columns=["iid", "appr_corr_0", "appr_corr_0.5_mat_0", "intr_corr_0", "intr_corr_0.5", "pos_corr_1.5", "pos_mutex_1.5", "seed"])

data_list = [ ["iid"], ["appr_corr_0", "appr_corr_0.5_mat_0"], ["intr_corr_0", "intr_corr_0.5"], ["pos_corr_1.5", "pos_mutex_1.5"] ]

# two datasets
r = 0
for comb in combinations(data_list, 2):
    for prod in product(comb[0], comb[1]):
        for seed in range(2):
            d1 = {prod[0]: 0.1, prod[1]:0.9, "seed": seed}
            d2 = {prod[0]: 0.5, prod[1]:0.5, "seed": seed}
            d3 = {prod[0]: 0.9, prod[1]:0.1, "seed": seed}
            df.at[str(r), :] = d1
            df.at[str(r+1), :] = d2
            df.at[str(r+2), :] = d3
            r +=3

# three datasets
for comb in combinations(data_list, 3):
    for prod in product(comb[0], comb[1], comb[2]):
        for seed in range(2):
            d1 = {prod[0]: 0.1, prod[1]:0.45, prod[2]:0.45, "seed": seed}
            d2 = {prod[0]: 0.45, prod[1]:0.1, prod[2]:0.45, "seed": seed}
            d3 = {prod[0]: 0.45, prod[1]:0.45, prod[2]:0.1, "seed": seed}
            d4 = {prod[0]: 0.33, prod[1]:0.34, prod[2]:0.33, "seed": seed}
            df.at[str(r), :] = d1
            df.at[str(r+1), :] = d2
            df.at[str(r+2), :] = d3
            df.at[str(r+3), :] = d4
            r += 4

# four datasets
for comb in combinations(data_list, 4):
    for prod in product(comb[0], comb[1], comb[2], comb[3]):
        for seed in range(2):
            d1 = {prod[0]: 0.1, prod[1]:0.3, prod[2]:0.3, prod[3]:0.3, "seed": seed}
            d2 = {prod[0]: 0.3, prod[1]:0.1, prod[2]:0.3, prod[3]:0.3, "seed": seed}
            d3 = {prod[0]: 0.3, prod[1]:0.3, prod[2]:0.1, prod[3]:0.3, "seed": seed}
            d4 = {prod[0]: 0.3, prod[1]:0.3, prod[2]:0.3, prod[3]:0.1, "seed": seed}
            d5 = {prod[0]: 0.25, prod[1]:0.25, prod[2]:0.25, prod[3]:0.25, "seed": seed}
            df.at[str(r), :] = d1
            df.at[str(r+1), :] = d2
            df.at[str(r+2), :] = d3
            df.at[str(r+3), :] = d4
            df.at[str(r+4), :] = d5
            r += 5

df.to_csv("/checkpoint/siruixie/data_mix.csv")
