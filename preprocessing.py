import logging
import torch
import ants
import tempfile
import os
from monai.transforms import (
    Compose, 
    LoadImaged, 
    EnsureChannelFirstd, 
    NormalizeIntensityd,
    EnsureTyped
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Template image path for registration (MNI305 space)
TEMPLATE_IMAGE_PATH = "average305_t1_tal_lin.nii"

# Define MONAI preprocessing pipeline
preprocessing = Compose([
    LoadImaged(keys="image"),
    EnsureChannelFirstd(keys="image"),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    EnsureTyped(keys="image")
])

def preprocess_image(image_path: str, segmentation_temp: str, device: torch.device = torch.device("cpu")) -> tuple[torch.Tensor, dict]:
    """Preprocess .nifti file with registration and MONAI transforms, return tensor and metadata."""
    try:
        # Load the source image
        logger.info(f"Loading source image: {image_path}")
        source_img = ants.image_read(image_path)

        # Registration to MNI305 space
        logger.info(f"Registering to MNI305 space: {TEMPLATE_IMAGE_PATH}")
        target_img = ants.image_read(TEMPLATE_IMAGE_PATH)
        registration = ants.registration(
            fixed=target_img,
            moving=source_img,
            type_of_transform="AffineFast"
        )
        registered_img = registration['warpedmovout']
        forward_transforms = registration['fwdtransforms']
        logger.info(f"Registration completed, forward transforms: {forward_transforms}")

        # Save registered image in segmentation_temp
        os.makedirs(segmentation_temp, exist_ok=True)
        registered_filename = os.path.basename(image_path).replace('.nii', '_registered.nii.gz')
        registered_path = os.path.join(segmentation_temp, registered_filename)
        ants.image_write(registered_img, registered_path)
        logger.info(f"Registered image saved: {registered_path}")

        # Move affine .mat file(s) to segmentation_temp
        new_forward_transforms = []
        for transform_path in forward_transforms:
            if transform_path.endswith('.mat'):
                new_transform_path = os.path.join(segmentation_temp, os.path.basename(transform_path))
                try:
                    os.rename(transform_path, new_transform_path)
                    logger.info(f"Affine transform moved to: {new_transform_path}")
                    new_forward_transforms.append(new_transform_path)
                except Exception as e:
                    logger.warning(f"Could not move affine transform: {transform_path} to {new_transform_path}: {e}")
                    new_forward_transforms.append(transform_path)
            else:
                new_forward_transforms.append(transform_path)

        # MONAI preprocessing
        logger.info("Starting MONAI preprocessing pipeline")
        data = {"image": registered_path}
        preprocessed_data = preprocessing(data)
        image_tensor = preprocessed_data["image"]
                
        # Log metadata
        logger.info(f"Preprocessing metadata keys: {list(preprocessed_data.get('image_meta_dict', {}).keys())}")
        logger.info(f"Transform history: {preprocessed_data.get('image_meta_dict', {}).get('image_transforms', 'N/A')}")
        logger.info(f"Normalization stats: {preprocessed_data.get('image_meta_dict', {}).get('norm_stats', 'N/A')}")
        logger.info(f"Affine: {preprocessed_data.get('image_meta_dict', {}).get('affine', 'N/A')}")
        
        # Ensure batch dimension
        if image_tensor.ndim == 4:  # [C, H, W, D]
            image_tensor = image_tensor.unsqueeze(0)  # [B, C, H, W, D]
        
        logger.info(f"Image preprocessed successfully: {image_path}, Shape: {image_tensor.shape}")
        return image_tensor.to(device), {
            "transforms": new_forward_transforms,
            "orig_image": image_path,
            "registered_image": registered_path
        }
    except Exception as e:
        logger.error(f"Preprocessing failed for {image_path}: {e}")
        raise