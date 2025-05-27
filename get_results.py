import os
import cv2
import numpy as np
#from skimage import metrics
import argparse
import mmcv

def calculate_iou_and_dice(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask)
    union = np.logical_or(pred_mask, gt_mask)

    #pred_mask = [[1 if element == 255 else element for element in row] for row in pred_mask]
    #gt_mask = [[1 if element == 255 else element for element in row] for row in gt_mask]
    iou = (np.sum(intersection)+1e-6) / (np.sum(union)+1e-6)

    pred_mask = np.asarray(pred_mask).astype(bool)
    gt_mask = np.asarray(gt_mask).astype(bool)

    # Compute Dice coefficient
    intersection = np.logical_and(pred_mask, gt_mask)

    dice = (2 * intersection.sum() +1e-6)/ (pred_mask.sum() + gt_mask.sum() +1e-6 )

    return iou, dice

def test(pred_path, gt_path, retour=False):
    global_iou = 0
    global_mean_iou = 0
    global_dice = 0
    global_mean_dice = 0
    total_samples = 0

    #print(os.listdir(gt_path))

    for gt_file in mmcv.scandir(gt_path,suffix=".png",recursive=True):
        
        # decomment this following if not in cityscapes dataset
        # if not gt_file.endswith('.png'):
        #     continue

        # Comment the following two lines if not cityscapes dataset
        if gt_file.endswith('_labelTrainIds.png'):
            continue

        gt_name = gt_file
        # Comment the following two lines if not cityscapes dataset
        
        gt_img = cv2.imread(os.path.join(gt_path, gt_name), cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(os.path.join(pred_path, gt_name), cv2.IMREAD_GRAYSCALE)
        pred[pred==90] = 0
        pred[pred==119] = 255

        # Calculate IoU and Dice score for each pair
        iou, dice = calculate_iou_and_dice(pred, gt_img)
        #mean_iou,mean_dice = calculate_mean_iou_and_dice(pred_thresholded, gt_img, class1_value=255, class2_value=0)

        print(f"GT: {gt_name}, iou: {iou}, dice: {dice}")

        # Accumulate global scores
        global_iou += iou
        #global_mean_iou += mean_iou
        global_dice += dice
        #global_mean_dice += mean_dice
        total_samples += 1

    if total_samples > 0:
        # Calculate global scores
        global_iou /= total_samples
        global_dice /= total_samples
        #global_mean_iou /= total_samples
        #global_mean_dice /= total_samples

        print(f"Global IoU: {global_iou}, Global mean IoU: {global_mean_iou}, Global Dice: {global_dice}, Global mean Dice: {global_mean_dice}")

    if retour :
        return global_iou

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate IoU and Dice scores for predicted masks compared to ground truth masks.")
    parser.add_argument("--pred_path", type=str, help="Path to the directory containing predicted masks.")
    parser.add_argument("--gt_path", type=str, help="Path to the directory containing ground truth masks.")
    args = parser.parse_args()

    test(args.pred_path, args.gt_path)