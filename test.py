import os

import torch
from monai.data import (DataLoader, Dataset, decollate_batch,
                        load_decathlon_datalist)
from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR
from torchprofile import profile_macs
from model_mvn.mvn import mvnNet
from monai.transforms import (Activationsd, AsDiscreted, Compose,
                              CropForegroundd, EnsureChannelFirstd,
                              EnsureTyped, Invertd, LoadImaged, Orientationd,
                              SaveImaged, ScaleIntensityRanged, Spacingd,
                              SpatialPadd)

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    # load validation dataset
    data_dir = ''
    datasets = os.path.join(data_dir, 'dataset.json')
    test_dataset_list = load_decathlon_datalist(datasets, is_segmentation=True, data_list_key='test')

    # define valid transform
    test_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Spacingd(keys=["image"],
                     pixdim=(0.2637, 0.2637, 0.8),
                     mode=("bilinear")),
            Orientationd(keys=["image"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=0,
                a_max=800,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image"], source_key="image"),
            SpatialPadd(keys=['image'], spatial_size=(64, 64, 64)),
        ]
    )

    # define post transform
    post_transforms = Compose(
        [
            EnsureTyped(keys="pred"),
            Activationsd(keys="pred", softmax=True),
            Invertd(
                keys="pred",
                transform=test_transforms,
                orig_keys="image",
                meta_keys="pred_meta_dict",
                orig_meta_keys="image_meta_dict",
                meta_key_postfix="meta_dict",
                nearest_interp=False,
                to_tensor=True,
            ),
            AsDiscreted(keys="pred", argmax=True),
            SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir="./predict_segmamba", output_postfix="seg",
                       resample=False),
        ]
    )

    test_dataset = Dataset(data=test_dataset_list, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=1)

    # load trained model for predict
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    patch_size = (64, 64, 64)

    model = mvnNet (
        in_chans=1,  # Number of input channels
        out_chans=2,  # Number of output classes
        feature_dims=[48, 96, 192, 384, 768],
    ).to(device)

    model.load_state_dict(torch.load(r"RESULTS/best_model.ckpt")['model'])
    model.eval()

    # Correctly create a dummy input tensor
    dummy_input = torch.randn((1, 1, 64, 64, 64)).to(device)  # Create a dummy input tensor

    # Calculate the FLOPs using the dummy input tensor
    flops = profile_macs(model, dummy_input)  # Pass the dummy input tensor
    print(f"Model FLOPs: {flops}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total trainable parameters: {total_params}')

    # start predcit and save predicted result to disk
    with torch.no_grad():
        for test_data in test_loader:
            test_inputs = test_data["image"].to(device)
            test_data['pred'] = sliding_window_inference(test_inputs, patch_size, 4, model)
            _ = [post_transforms(i) for i in decollate_batch(test_data)]
