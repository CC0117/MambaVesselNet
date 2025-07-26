import os
from argparse import ArgumentParser
import numpy as np
import torch
import datetime
from torch.utils.tensorboard import SummaryWriter
from loguru import logger
from monai.data import (CacheDataset, ThreadDataLoader, decollate_batch,
                        load_decathlon_datalist)
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.transforms import (AsDiscrete, Compose, CropForegroundd,
                              EnsureChannelFirstd, LoadImaged, Orientationd,
                              RandAffined, RandCropByPosNegLabeld, SpatialPadd,
                              RandShiftIntensityd, ScaleIntensityRanged,
                              Spacingd, ToTensord)
from monai.utils import set_determinism
from model_mvn.mvn import mvnNet
os.environ['CUDA_VISIBLE_DEVICES'] = "1"


def run_train(args):
    # set random seed
    set_determinism(seed=args['seed'])

    # Using date and hour
    current_date_hour = datetime.datetime.now().strftime("%Y%m%d-%H")
    log_dir = os.path.join("runs", "training_" + current_date_hour)
    writer = SummaryWriter(log_dir)

    # use device for train
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    print(torch.cuda.is_available())

    # load dataset for train and valid
    datasets = args['dataset']
    train_dataset_list = load_decathlon_datalist(datasets, is_segmentation=True, data_list_key='training')
    valid_dataset_list = load_decathlon_datalist(datasets, is_segmentation=True, data_list_key='validation')

    # define dataset train and valid transform
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"],
                 pixdim=(0.2637, 0.2637, 0.8),
                 mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=0,
            a_max=800,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadd(keys=['image', 'label'], spatial_size=(64, 64, 48)),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(64, 64, 64),
            pos=1,
            neg=1,
            num_samples=4,
            image_key="image",
            image_threshold=0,
        ),
        RandShiftIntensityd(
            keys=["image"],
            offsets=0.10,
            prob=0.50,
        ),
        RandAffined(keys=['image', 'label'],
                    mode=('bilinear', 'nearest'),
                    prob=1.0,
                    spatial_size=(64, 64, 64),
                    rotate_range=(0, 0, np.pi / 30),
                    scale_range=(0.1, 0.1, 0.1)),
        ToTensord(keys=["image", "label"]),
    ])

    valid_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"],
                 pixdim=(0.2637, 0.2637, 0.8),
                 mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=0,
            a_max=800,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadd(keys=['image', 'label'], spatial_size=(64, 64, 64)),
        ToTensord(keys=["image", "label"]),
    ])

    # start loading data from disk
    train_dataset = CacheDataset(train_dataset_list, transform=train_transforms, cache_num=30, cache_rate=1.0,
                                 num_workers=args['num_workers'])
    valid_dataset = CacheDataset(valid_dataset_list, transform=valid_transforms, cache_num=5, cache_rate=1.0,
                                 num_workers=args['num_workers'])

    train_loader = ThreadDataLoader(train_dataset, num_workers=0, batch_size=args['batch_size'], shuffle=True)
    valid_loader = ThreadDataLoader(valid_dataset, num_workers=0, batch_size=1, shuffle=False)

    # define vnet network, no drop out
    patch_size = (64, 64, 64)

    # Initialize SegMamba model
    model = mvnNet (
        in_chans=1,  # Number of input channels
        out_chans=args['num_classes'],  # Number of output classes
        feature_dims=[48, 96, 192, 384, 768],
    ).to(device)

    # define loss function, multi class use dice + ce
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=1e-5)

    # some hyber-params
    max_iterations = args['max_iter']
    valid_iter = args['valid_iter']
    post_label = AsDiscrete(to_onehot=args['num_classes'])
    post_pred = AsDiscrete(argmax=True, to_onehot=args['num_classes'])
    dice_metric = DiceMetric(include_background=False, reduction='mean', get_not_nans=False)
    global_step = args['epoch']
    dice_valid_best = 0
    global_step_best = 0
    epoch_loss_values = []
    metric_values = []

    # define learning rate adjust scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iterations, eta_min=1e-7, last_epoch=-1)

    # start training....
    while global_step < max_iterations:
        # train
        model.train()
        epoch_loss = 0
        step = 0
        for step, batch in enumerate(train_loader):
            step += 1
            x, y = batch['image'].to(device), batch['label'].to(device)
            logit_map = model(x)
            loss = loss_function(logit_map, y)

            # log the loss into tensorboard
            writer.add_scalar('Loss/train', loss.item(), global_step)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            logger.info(
                "Epoch:[{:0>5d}/{:0>5d}]\t train_loss = {:.5f}\t".format(global_step, max_iterations, loss.item()))

            optimizer.zero_grad()

            # online validation
            if (global_step % valid_iter == 0 and global_step != 0) or (global_step == max_iterations):
                model.eval()
                with torch.no_grad():
                    for step, batch in enumerate(valid_loader):
                        valid_in, target = batch['image'].to(device), batch['label'].to(device)

                        # sliding window inference according to patch size, batch is 4
                        valid_out = sliding_window_inference(valid_in, patch_size, 4, model)

                        valid_labels_list = decollate_batch(target)
                        valid_labels_convert = [post_label(valid_label_tensor) for valid_label_tensor in
                                                valid_labels_list]

                        valid_output_list = decollate_batch(valid_out)
                        valid_output_convert = [post_pred(valid_pred_tensor) for valid_pred_tensor in valid_output_list]
                        dice_metric(y_pred=valid_output_convert, y=valid_labels_convert)

                    mean_dice_val = dice_metric.aggregate().item()
                    writer.add_scalar('Metric/Validation Dice', mean_dice_val, global_step)
                    dice_metric.reset()

                    epoch_loss /= step
                    writer.add_scalar('Loss/Epoch', epoch_loss / step, global_step)
                    epoch_loss_values.append(epoch_loss)
                    metric_values.append(mean_dice_val)

                    logger.info("valid step: {:0>5d} mean dice = {:.8f}".format(global_step, mean_dice_val))

                    # saved model
                    if mean_dice_val > dice_valid_best:
                        dice_valid_best = mean_dice_val
                        global_step_best = global_step
                        state_dict = {
                            'model': model.state_dict(),
                            'dice_metric': dice_valid_best,
                            'epoch': global_step_best,
                        }

                        torch.save(state_dict, os.path.join(args['checkpoint_dir'], 'best_model.ckpt'))
                        logger.info(
                            f"model was saved! current best Avg. Dice = {dice_valid_best}, current Avg. Dice = {mean_dice_val} at {global_step_best} iteration.")
                    else:
                        logger.info(
                            f"model was not saved! current best Avg. Dice = {dice_valid_best}, current Avg. Dice = {mean_dice_val}")

            global_step += 1
            scheduler.step()
    writer.close()


if __name__ == "__main__":
    parser = ArgumentParser(description="3d volume segmentation")
    parser.add_argument("--dim_in", type=int, default=1, help="input dimension")
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--max_iter", type=int, default=5000, help="maximum number of iterations to train")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="number of workers for data loading")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--num_classes", type=int, default=2, help="number of classes")
    parser.add_argument("--num_samples", type=int, default=4, help="number of samples to use for training")
    parser.add_argument("--valid_iter", type=int, default=500, help="number of iterations to validate")
    parser.add_argument("--dataset", type=str, default="dataset.json")
    parser.add_argument("--pretrain_weights", type=str, default="./RESULTS/model_best.pt")
    parser.add_argument("--checkpoint_dir", type=str, default='./RESULTS')
    args = parser.parse_args()

    # create model saved dir
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    # train and valid log for check
    logger_file = "train_Segmamba.log"
    logger.add(logger_file, rotation="50 MB")

    run_train(vars(args))