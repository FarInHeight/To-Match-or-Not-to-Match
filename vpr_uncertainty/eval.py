import numpy as np
import os, argparse, sys
from sklearn.metrics import precision_recall_curve, auc
from glob import glob
from pathlib import Path
import torch

sys.path.append(str(
    Path(__file__).parent.parent
))

from util import get_list_distances_from_preds
from vpr_uncertainty.baselines import compute_l2, compute_pa, compute_sue, compute_random

np.random.seed(4082025)

def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--preds-dir", type=str, help="directory with predictions of a VPR model")
    parser.add_argument("--inliers-dir", type=str, help="directory with image matching results")
    parser.add_argument("--z-data-path", type=str, help="path to the 'z_data' file (output part of VPR evaluation)")
    parser.add_argument(
        "--positive-dist-threshold",
        type=int,
        default=25,
        help="distance (in meters) for a prediction to be considered a positive",
    )

    return parser.parse_args()

def main(args):
    preds_folder = args.preds_dir
    inliers_folder = Path(args.inliers_dir)
    threshold = args.positive_dist_threshold
    z_data_path = args.z_data_path

    txt_files = glob(os.path.join(preds_folder, "*.txt"))
    txt_files.sort(key=lambda x: int(Path(x).stem))

    dict_results = torch.load(z_data_path, weights_only=False)
    ref_poses = dict_results['database_utms']
    preds = dict_results['predictions']
    dists = dict_results['distances']

    total_queries = len(txt_files)
    matched_array_for_aucpr = np.zeros(total_queries, dtype="float32")
    inliers_scores = np.zeros(total_queries, dtype="float32")

    for itr, txt_file_query in enumerate(txt_files):
        geo_dists = get_list_distances_from_preds(txt_file_query)
        matched_array_for_aucpr[itr]=1.0 if geo_dists[0] <= threshold else 0.0 #checking if Top-1 contains GT

        torch_file_query = inliers_folder.joinpath(Path(txt_file_query).name.replace('txt', 'torch'))
        query_inliers_results = torch.load(torch_file_query, weights_only=False)
        inliers_scores[itr] = query_inliers_results[0]['num_inliers']
    
    inliers_scores = np.interp(inliers_scores, (inliers_scores.min(), inliers_scores.max()), (0.0, 1.0))
    precision_based_on_inliers, recall_based_on_inliers, _ = precision_recall_curve(matched_array_for_aucpr, inliers_scores)
    auc_based_on_inliers = auc(recall_based_on_inliers, precision_based_on_inliers)

    auc_based_on_l2 = compute_l2(matched_array_for_aucpr, dists)
    auc_based_on_pa = compute_pa(matched_array_for_aucpr, dists)
    auc_based_on_sue = compute_sue(matched_array_for_aucpr, preds, ref_poses, dists, num_NN=10, slope=350)
    auc_based_on_random = compute_random(matched_array_for_aucpr)

    print(f'L2-distance: {auc_based_on_l2 * 100:.1f}')
    print(f'PA-score: {auc_based_on_pa * 100:.1f}')
    print(f'SUE: {auc_based_on_sue * 100:.1f}')
    print(f'Random: {auc_based_on_random * 100:.1f}')
    print(f'Inliers: {auc_based_on_inliers * 100:.1f}')

if __name__ == "__main__":
    args = parse_arguments()
    main(args)