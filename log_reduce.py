# A python script to collect all relevant logs and store in a .csv file
import os
import json
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
    for i in range(num_entries):
        path = "/checkpoint/siruixie/runs/objectness/hydra_train_ds_mix/data_mix_idx="+str(i)+",sweep_name=ds_mix/wandb/"
        json_path = ""
        count = 0
        for dir_name in os.listdir(path):
            if dir_name.startswith("run-"+str(date)):
                json_path = os.path.join(path, dir_name, "files/wandb-summary.json")
                count += 1
        assert count <= 1, "Multiple experiments on this date!"
        # assert not len(json_path) == 0, "No experiment on this date for index "+str(i)
        if len(json_path) == 0:
            pass
        else:
            data_weights = df.iloc[i+1, 1:]
            print(data_weights)

            with open(json_path, 'r') as f:
                logged_metrics = json.load(f)

                del logged_metrics["images"]
                del logged_metrics["_runtime"]
                del logged_metrics["_timestamp"]
                del logged_metrics["_step"]
                del logged_metrics["lr-Adam"]
                del logged_metrics["epoch"]
                row = {**data_weights, **logged_metrics}
                print(row)
                if result is not None:
                    result.at[str(i), :] = row
                else:
                    result = pd.DataFrame(columns=list(row.keys()))
                    result.at[str(i), :] = row
    result.to_csv(result_csv)

if __name__ == '__main__':
    args = parser.parse_args()
    reduce_logs(args.date, args.num_exp)
