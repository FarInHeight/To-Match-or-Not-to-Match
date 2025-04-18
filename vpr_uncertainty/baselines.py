import math
import numpy as np
from tqdm import tqdm, trange
from sklearn.metrics import precision_recall_curve, auc

def compute_random(matched_array_for_aucpr):
    total_queries = len(matched_array_for_aucpr)
    random_scores = np.random.uniform(size=total_queries)
    random_scores = np.interp(random_scores, (random_scores.min(), random_scores.max()), (0.0, 1.0))
    precision_based_on_random, recall_based_on_random, _ = precision_recall_curve(matched_array_for_aucpr, random_scores)

    return auc(recall_based_on_random, precision_based_on_random)

def compute_l2(matched_array_for_aucpr, dists):
    total_queries = len(matched_array_for_aucpr)
    l2distances = np.zeros(total_queries)

    for itr in trange(total_queries, desc='Computing L2 scores'):
        l2distances[itr] = dists[itr][0]
    
    l2_scores = -1 * l2distances
    l2_scores = np.interp(l2_scores, (l2_scores.min(), l2_scores.max()), (0.1, 1.0))
    precision_based_on_l2, recall_based_on_l2, _ = precision_recall_curve(matched_array_for_aucpr, l2_scores)

    return auc(recall_based_on_l2, precision_based_on_l2)

def compute_pa(matched_array_for_aucpr, dists):
    total_queries = len(matched_array_for_aucpr)
    pa_scores = np.zeros(total_queries)
    
    for itr in trange(total_queries, desc='Computing PA scores'):
        pa_scores[itr] =  dists[itr][0] / dists[itr][1]  
        
    pa_scores = -1 * pa_scores    
    pa_scores = np.interp(pa_scores, (pa_scores.min(), pa_scores.max()), (0.1, 1.0))
    precision_based_on_pa, recall_based_on_pa, _ = precision_recall_curve(matched_array_for_aucpr, pa_scores)

    return auc(recall_based_on_pa, precision_based_on_pa)

def compute_sue(matched_array_for_aucpr, preds, ref_poses, dists, num_NN=10, slope=350):
    # num_NN: Number of nearest neighbours 
    # slope: Slope hyper-parameter of the Gaussian used to down-weight the contributions of nearest neighbours 
    total_queries = len(matched_array_for_aucpr)
    sue_scores = np.zeros(total_queries)

    weights = np.ones(num_NN)
    for itr in tqdm(range(len(sue_scores)), desc='Computing SUE scores'):   
        top_preds = preds[itr][:num_NN]
        nn_poses = ref_poses[top_preds]
        
        for itr2 in range(num_NN):
            weights[itr2] = math.e ** ((-1*abs(dists[itr][itr2])) * slope) 

        weights = weights/sum(abs(weights))

        mean_pose = np.asarray([np.average(nn_poses[:,0], weights=weights), np.average(nn_poses[:,1], weights=weights)])

        variance_lat_lat = 0 
        variance_lon_lon = 0    
        variance_lat_lon = 0    

        for k in range(0, num_NN):                
            diff_lat_lat = min(500, nn_poses[k,0] - mean_pose[0]) # so everything that is more than 500 meters away contributes equally to the variance 
            diff_lon_lon = min(500, nn_poses[k,1] - mean_pose[1])
            diff_lat_lon = min(500, nn_poses[k,0] - mean_pose[0]) *  min(500, nn_poses[k,1] - mean_pose[1])
                    
            variance_lat_lat = variance_lat_lat + weights[k] * (diff_lat_lat)**2
            variance_lon_lon = variance_lon_lon + weights[k] * (diff_lon_lon)**2
            variance_lat_lon = variance_lat_lon + weights[k] * diff_lat_lon
            
        sue_scores[itr] = (variance_lat_lat + variance_lon_lon)/2  # assuming independent dimensions

    sue_scores = -1 * sue_scores # converting into a confidence instead of an uncertainty
    sue_scores_normalized = np.interp(sue_scores, (sue_scores.min(), sue_scores.max()), (0.0, 0.9999)) # avoiding infinity

    precision_based_on_suescore, recall_based_on_suescore, _ = precision_recall_curve(matched_array_for_aucpr, sue_scores_normalized)
    return auc(recall_based_on_suescore, precision_based_on_suescore)