# A python script to collect all relevant logs and store in a .csv file
import os
import json
import ast
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--date', help="date of the experiments")
parser.add_argument('--num_exp', default=1, type=int, help="The number of experiments to collect.")

def reduce_logs(date, num_entries):
    result_csv = "/checkpoint/siruixie/data_mix_result_{}.csv".format(date)
    df = pd.read_csv("/checkpoint/siruixie/data_mix.csv")
    if os.path.exists(result_csv):
        result = pd.read_csv(result_csv)
    else:
        result = None
    for i in range(1, num_entries):
        path = "/checkpoint/siruixie/runs/objectness/hydra_test_test1/data_mix_idx="+str(i)+",lr=0.0002,num_iterations=4,num_train_images=500,sweep_name=test1/wandb/latest-run/files/output.log"

        data_weights = df.iloc[i, 1:]
        # print(data_weights)
        with open(path, 'r') as f:
            for line in reversed(f.readlines()):
                if line[:8] == "{'epoch'":
                    logged_metrics = line.replace("tensor([", "")
                    logged_metrics = logged_metrics.replace("tensor(", "")
                    logged_metrics = logged_metrics.replace("], device='cuda:0')", "")
                    logged_metrics = logged_metrics.replace(", device='cuda:0')", "")
                    logged_metrics = logged_metrics.replace("'", '"')
                    print(logged_metrics)
                    logged_metrics = json.loads(logged_metrics)

                    row = {**data_weights, **logged_metrics}
                    # print(row)
                    if result is not None:
                        result.at[str(i), :] = row
                    else:
                        result = pd.DataFrame(columns=list(row.keys()))
                        result.at[str(i), :] = row
    result.to_csv(result_csv)

if __name__ == '__main__':
    args = parser.parse_args()
    reduce_logs(args.date, args.num_exp)
