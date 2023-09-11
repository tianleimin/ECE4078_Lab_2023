import json
import numpy as np
from copy import deepcopy

# list of target fruit and veg types
TARGET_TYPES = ['orange', 'lemon', 'lime', 'tomato', 'capsicum', 'potato', 'pumpkin', 'garlic']
#TARGET_TYPES = [target.upper() for target in TARGET_TYPES]

# read ground-truth map containing both ARUCO and target poses
def parse_true_map(fname):
    with open(fname, 'r') as fd:
        gt_dict = json.load(fd)
    aruco_dict = {}
    target_dict = {}

    for key in gt_dict:
        # read SLAM map
        if key.startswith('aruco'):
            aruco_num = int(key.strip('aruco')[:-2])
            aruco_dict[aruco_num] = np.reshape([gt_dict[key]['x'], gt_dict[key]['y']], (2, 1))
        # read target map
        else:
            object_type = key.split('_')[0]
            if object_type not in target_dict:
                target_dict[object_type] = np.array([[gt_dict[key]['x'], gt_dict[key]['y']]])
            else:
                target_dict[object_type] = np.append(target_dict[object_type], [[gt_dict[key]['x'], gt_dict[key]['y']]], axis=0)
                
    #print(f'True marker poses: {aruco_dict}')
    #print(f'True target poses: {target_dict}')

    return aruco_dict, target_dict

# read estimated SLAM map 'lab_output/slam.txt'
def parse_slam_map(fname: str) -> dict:
    with open(fname, 'r') as fd:
        usr_dict = json.load(fd)
    aruco_dict = {}
    for (i, tag) in enumerate(usr_dict['taglist']):
        aruco_dict[tag] = np.reshape([usr_dict['map'][0][i], usr_dict['map'][1][i]], (2, 1))
    
    #print(f'Estimated marker poses: {aruco_dict}')
    
    return aruco_dict

# read estimated target map 'lab_output/targets.txt'
def parse_object_map(fname):
    with open(fname, 'r') as fd:
        usr_dict = json.load(fd)
    target_dict = {}

    for key in usr_dict:
        object_type = key.split('_')[0]
        if object_type not in target_dict:
            target_dict[object_type] = np.array([[usr_dict[key]['x'], usr_dict[key]['y']]])
        else:
            target_dict[object_type] = np.append(target_dict[object_type], [[usr_dict[key]['x'], usr_dict[key]['y']]], axis=0)

    #print(f'Estimated target poses: {target_dict}')
                    
    return target_dict

####################################
# for SLAM evaluation
def match_aruco_points(aruco0: dict, aruco1: dict):
    points0 = []
    points1 = []
    keys = []
    for key in aruco0:
        if not key in aruco1:
            continue

        points0.append(aruco0[key])
        points1.append(aruco1[key])
        keys.append(key)
        
    return keys, np.hstack(points0), np.hstack(points1)


def solve_umeyama2d(points1, points2):
    # Solve the optimal transform such that
    # R(theta) * p1_i + t = p2_i

    assert (points1.shape[0] == 2)
    assert (points1.shape[0] == points2.shape[0])
    assert (points1.shape[1] == points2.shape[1])

    # Compute relevant variables
    num_points = points1.shape[1]
    mu1 = 1 / num_points * np.reshape(np.sum(points1, axis=1), (2, -1))
    mu2 = 1 / num_points * np.reshape(np.sum(points2, axis=1), (2, -1))
    sig1sq = 1 / num_points * np.sum((points1 - mu1) ** 2.0)
    sig2sq = 1 / num_points * np.sum((points2 - mu2) ** 2.0)
    Sig12 = 1 / num_points * (points2 - mu2) @ (points1 - mu1).T

    # Use the SVD for the rotation
    U, d, Vh = np.linalg.svd(Sig12)
    S = np.eye(2)
    if np.linalg.det(Sig12) < 0:
        S[-1, -1] = -1

    # Return the result as an angle and a 2x1 vector
    R = U @ S @ Vh
    theta = np.arctan2(R[1, 0], R[0, 0])
    x = mu2 - R @ mu1

    return theta, x

def apply_transform(theta, x, points):
    # Apply an SE(2) transform to a set of 2D points
    assert (points.shape[0] == 2)

    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))

    points_transformed = R @ points + x
    
    return points_transformed

def compute_slam_rmse(points1, points2):
    # Compute the RMSE between two matched sets of 2D points.
    assert (points1.shape[0] == 2)
    assert (points1.shape[0] == points2.shape[0])
    assert (points1.shape[1] == points2.shape[1])
    num_points = points1.shape[1]
    residual = (points1 - points2).ravel()
    MSE = 1.0 / num_points * np.sum(residual ** 2)

    return np.sqrt(MSE)

####################################
# for target pose estimation evaluation
def compute_object_est_error(gt_list, est_list):
    """Compute the object target pose estimation error based on Euclidean distance

    If there are more estimations than the number of targets (e.g. only 1 target orange, but detected 2),
        then take the average error of the 2 detections

    if there are fewer estimations than the number of targets (e.g. 2 target oranges, but only detected 1),
        then return [MAX_ERROR, error with the closest target]

    @param gt_list: target ground truth list
    @param est_list: estimation list
    @return: error of all the objects
    """

    MAX_ERROR = 1

    object_errors = {}

    for target_type in gt_list:
        n_gt = len(gt_list[target_type])  # number of targets in this fruit type

        type_errors = []
        for i, gt in enumerate(gt_list[target_type]):
            dist = []
            try:
                for est in est_list[target_type]:
                    dist.append(np.linalg.norm(gt - est))  # compute Euclidean distance
    
                n_est = len(est_list[target_type])
    
                # if this fruit type has been detected
                if len(dist) > 0:
                    if n_est > n_gt:    # if more estimation than target, take the mean error
                        object_errors[target_type + '_{}'.format(i)] = np.round(np.mean(dist), 3)
                    elif n_est < n_gt:  # see below
                        type_errors.append(np.min(dist))
                    else:   # for normal cases, n_est == n_gt, take the min error
                        object_errors[target_type + '_{}'.format(i)] = np.round(np.min(dist), 3)
            except:   # if there is no estimation for this fruit type
                for j in range(n_gt):
                    object_errors[target_type + '_{}'.format(j)] = MAX_ERROR

        if len(type_errors) > 0:    # for the n_est < n_gt scenario
            type_errors = np.sort(type_errors)
            for i in range(len(type_errors) - 1):
                object_errors[target_type + '_{}'.format(i+1)] = np.round(type_errors[i], 3)
            object_errors[target_type + '_{}'.format(0)] = MAX_ERROR



    return object_errors

def align_object_poses(theta, x, objects_est):
    objects = deepcopy(objects_est)

    for object in objects:
        poses = []
        for pos in objects[object]:
            pos = np.reshape(pos, (2, 1))
            pos = apply_transform(theta, x, pos)
            pos = np.reshape(pos, (1, 2))[0]

            poses.append(pos)

        objects[object] = poses

    return objects

####################################
# main loop
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser('Matching the estimated map and the true map')
    parser.add_argument('--true-map', type=str, default='true_map.txt')
    parser.add_argument('--slam-est', type=str, default='lab_output/slam.txt')
    parser.add_argument('--target-est', type=str, default='lab_output/targets.txt')
    parser.add_argument('--slam-only', action='store_true')
    parser.add_argument('--target-only', action='store_true')
    args, _ = parser.parse_known_args()

    slam_only = args.slam_only      # only evaluate slam.txt
    target_only = args.target_only  # only evaluate targets.txt

    if slam_only and target_only:
        print('You cannot choose --slam-only and --target-only at the same time!')
        exit()

    aruco_gt, objects_gt = parse_true_map(args.true_map)

    slam_rmse = 99

    if slam_only:
        # only evaluate SLAM
        aruco_est = parse_slam_map(args.slam_est)
        taglist, slam_est_vec, slam_gt_vec = match_aruco_points(aruco_est, aruco_gt)
        theta, x = solve_umeyama2d(slam_est_vec, slam_gt_vec)
        slam_est_vec_aligned = apply_transform(theta, x, slam_est_vec)

        slam_rmse = compute_slam_rmse(slam_est_vec_aligned, slam_gt_vec)

        print(f'The SLAM RMSE = {np.round(slam_rmse, 3)}')

    elif target_only:
        objects_est = parse_object_map(args.target_est)
        object_est_errors = compute_object_est_error(objects_gt, objects_est)
        print('Object pose estimation errors:')
        print(json.dumps(object_est_errors, indent=4))
    else:
        # evaluate SLAM
        aruco_est = parse_slam_map(args.slam_est)
        taglist, slam_est_vec, slam_gt_vec = match_aruco_points(aruco_est, aruco_gt)
        theta, x = solve_umeyama2d(slam_est_vec, slam_gt_vec)
        slam_est_vec_aligned = apply_transform(theta, x, slam_est_vec)

        slam_rmse_raw = compute_slam_rmse(slam_est_vec, slam_gt_vec)
        slam_rmse_aligned = compute_slam_rmse(slam_est_vec_aligned, slam_gt_vec)

        print(f'The SLAM RMSE before alignment = {np.round(slam_rmse_raw, 3)}')
        print(f'The SLAM RMSE after alignment = {np.round(slam_rmse_aligned, 3)}')

        print('----------------------------------------------')
        # evaluate object pose estimation errors
        objects_est = parse_object_map(args.target_est)

        # align the object poses using the transform computed from SLAM
        objects_est_aligned = align_object_poses(theta, x, objects_est)

        object_est_errors_raw = compute_object_est_error(objects_gt, objects_est)
        object_est_errors_aligned = compute_object_est_error(objects_gt, objects_est_aligned)

        print('Object pose estimation errors before alignment:')
        print(json.dumps(object_est_errors_raw, indent=4))
        print('Object pose estimation errors after alignment:')
        print(json.dumps(object_est_errors_aligned, indent=4))

        err_lst = []
        target_est_score = 0
        target_scores = []
        for object_err in object_est_errors_aligned:
            est_err = object_est_errors_aligned[object_err]
            if est_err > 1.0:
                target_score = 0
            elif est_err <= 0.025:
                target_score = 8
            else:
                target_score = (1.0-est_err)/(1-0.025) * 8.0
            err_lst.append(est_err)
            target_scores.append(target_score)

        print(f'Average object pose estimation error after alignment: {np.mean(err_lst)}')
        print(f'Individual target scores: {target_scores}')
        print('======')
        print(f'Target Estimation Error Score (0 to 80): = {np.sum(target_scores)}')





