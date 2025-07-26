import json
import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import datetime
from monai.metrics import DiceMetric, MeanIoU
from monai.transforms import AsDiscrete, Compose, EnsureType


def calculate_metrics(pred, true):
    TP = np.sum((pred == 1) & (true == 1))
    FP = np.sum((pred == 1) & (true == 0))
    TN = np.sum((pred == 0) & (true == 0))
    FN = np.sum((pred == 0) & (true == 1))

    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    Vp = np.sum(pred == 1)
    Vt = np.sum(true == 1)
    volume_similarity = 1 - abs(Vp - Vt) / (Vp + Vt - abs(Vp - Vt)) if (Vp + Vt - abs(Vp - Vt)) > 0 else 0

    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "volume_similarity": volume_similarity
    }


def main():
    pred_dir = "./predict_segmamba"
    gt_dir = "/home/Yanming_Chen/Json_dataset/labelsTr"
    dataset_json = "dataset.json"

    with open(dataset_json, "r") as f:
        test_data = json.load(f)['test']

    dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
    mean_iou_metric = MeanIoU(include_background=False, ignore_empty=True)

    post_transforms = Compose([EnsureType(), AsDiscrete(threshold_values=True)])

    eval_dict = {}
    total_dice, total_iou = 0, 0
    total_sensitivity, total_specificity, total_precision, total_volume_similarity = 0, 0, 0, 0
    for item in test_data:
        base_name = os.path.basename(item['image']).split('.')[0]
        pred_image = os.path.join(pred_dir, base_name, f"{base_name}_seg.nii.gz")
        true_image = os.path.join(gt_dir, f"{base_name}_GT.nii.gz")

        sitk_pred = sitk.ReadImage(pred_image)
        sitk_true = sitk.ReadImage(true_image)
        pred_array = sitk.GetArrayFromImage(sitk_pred)
        true_array = sitk.GetArrayFromImage(sitk_true)

        assert pred_array.shape == true_array.shape, f"{pred_image} and {true_image} do not match in size."

        pred_array = post_transforms(pred_array)
        true_array = post_transforms(true_array)

        dice_metric(y_pred=pred_array[None, None], y=true_array[None, None])
        mean_iou_metric(y_pred=pred_array[None], y=true_array[None])

        dice_score = dice_metric.aggregate().item()
        dice_metric.reset()
        mean_iou_score = mean_iou_metric.aggregate().item()
        mean_iou_metric.reset()

        additional_metrics = calculate_metrics(pred_array, true_array)
        eval_dict[base_name] = {
            "DICE": dice_score,
            "Mean IoU": mean_iou_score,
            **additional_metrics
        }
        print(
            f"Evaluation {base_name}, DICE={dice_score:.5f}, Mean IoU={mean_iou_score:.5f}, Sensitivity={additional_metrics['sensitivity']:.5f}, Specificity={additional_metrics['specificity']:.5f}, Precision={additional_metrics['precision']:.5f}, Volume Similarity={additional_metrics['volume_similarity']:.5f}")

        total_dice += dice_score
        total_iou += mean_iou_score
        total_sensitivity += additional_metrics['sensitivity']
        total_specificity += additional_metrics['specificity']
        total_precision += additional_metrics['precision']
        total_volume_similarity += additional_metrics['volume_similarity']

    num_entries = len(test_data)
    mean_dice = total_dice / num_entries
    mean_iou = total_iou / num_entries
    mean_sensitivity = total_sensitivity / num_entries
    mean_specificity = total_specificity / num_entries
    mean_precision = total_precision / num_entries
    mean_volume_similarity = total_volume_similarity / num_entries

    print(f"Overall Mean DICE: {mean_dice:.5f}")
    print(f"Overall Mean IoU: {mean_iou:.5f}")
    print(f"Overall Mean Sensitivity: {mean_sensitivity:.5f}")
    print(f"Overall Mean Specificity: {mean_specificity:.5f}")
    print(f"Overall Mean Precision: {mean_precision:.5f}")
    print(f"Overall Mean Volume Similarity: {mean_volume_similarity:.5f}")

    eval_dict["Overall Mean"] = {
        "DICE": mean_dice,
        "Mean IoU": mean_iou,
        "Sensitivity": mean_sensitivity,
        "Specificity": mean_specificity,
        "Precision": mean_precision,
        "Volume Similarity": mean_volume_similarity,
    }

    df = pd.DataFrame.from_dict(eval_dict, orient="index")
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H")
    csv_filename = f"evaluation_segnet_{current_time}_hours.csv"
    df.to_csv(csv_filename, index=True, header=True)
    print(f"Results saved to {csv_filename}")


if __name__ == "__main__":
    main()