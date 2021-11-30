import numpy as np
import pandas as pd
import torch
from collections import defaultdict

from eval_utils import eval_all_metrics, SUBSET2IDS

ROOT = ""  # the folder which you place the downloaded scores

"""
organize the evaluation results in a table
"""
def get_allresults_df(root: str) -> pd.DataFrame:
    data_dict = defaultdict(list)
    # these are the three baseline model ablations
    for model_type in ["image", "image_cam", "image_hs_cam"]:
        data_dict["model"].append(model_type)
        val_f1s = []  # 5 x 28
        all_f1s = defaultdict(list)

        # get results for each run
        for run_num in range(5):
            d_dict = torch.load(f"{root}/{model_type}_{run_num}.pth")
            f1_dict = eval_all_metrics(
                d_dict["val_scores"], d_dict["test_scores"],
                d_dict["val_targets"], d_dict["test_targets"]
            )
            for k, v in f1_dict.items():
                if isinstance(v, float):
                    all_f1s[k].append(v * 100)
                else:
                    all_f1s[k].append(np.array(v)[np.newaxis, :] * 100)

        val_f1s = np.vstack(all_f1s["val_none"])

        for e_type, c_ids in SUBSET2IDS.items():
            e_f1s = np.mean(np.hstack([val_f1s[:, c:c+1] for c in c_ids]), 1)
            data_dict[f"val-{e_type}"].append("{:.2f} +- {:.2f}".format(
                np.mean(e_f1s), np.std(e_f1s)
            ))

        for k, values in all_f1s.items():
            if not k.endswith("none"):
                data_dict[k].append("{:.2f} +- {:.2f}".format(
                    np.mean(values), np.std(values)
                ))
    df = pd.DataFrame(data_dict)
    return df


# these are the same results reported in README.md
get_allresults_df(ROOT)