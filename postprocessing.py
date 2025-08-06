import os
import torch
import logging
import nibabel as nib
import ants
from monai.transforms import Compose, Activationsd, AsDiscreted, KeepLargestConnectedComponentd
import numpy as np

logger = logging.getLogger("postprocessing")
logging.basicConfig(level=logging.INFO)

postprocessing = Compose([
    Activationsd(keys="pred", softmax=True, dim=1),
    AsDiscreted(keys="pred", argmax=True, dim=1),
    KeepLargestConnectedComponentd(keys="pred")
])

def postprocess_output(pred_output: torch.Tensor, meta_dict: dict, mask_path: str, output_dir: str, output_name: str) -> str:
    logger.info("Starting postprocessing...")
    logger.info(f"Input prediction shape: {pred_output.shape}, type: {type(pred_output)}")
    logger.info(f"Meta dict keys: {list(meta_dict.keys())}")
    logger.info(f"Mask path: {mask_path}, output_dir: {output_dir}, output_name: {output_name}")

    try:
        # MONAI postprocessing
        logger.info("Applying MONAI postprocessing transforms...")
        input_data = {"pred": pred_output}
        postprocessed_data = postprocessing(input_data)
        pred_tensor = postprocessed_data["pred"]
        logger.info(f"After MONAI postprocessing, pred shape: {pred_tensor.shape}")

        # Ensure 3D tensor
        if pred_tensor.ndim == 5:  # [B, C, H, W, D]
            pred_tensor = pred_tensor.squeeze(0).squeeze(0)
        elif pred_tensor.ndim == 4:  # [B, H, W, D] or [C, H, W, D]
            pred_tensor = pred_tensor.squeeze(0)
        logger.info(f"Postprocessed tensor after squeeze shape: {pred_tensor.shape}")

        # Load registered and original images
        registered_image_path = meta_dict.get("registered_image")
        orig_image_path = meta_dict.get("orig_image")
        if not registered_image_path or not os.path.exists(registered_image_path):
            raise FileNotFoundError(f"Registered image not found: {registered_image_path}")
        if not orig_image_path or not os.path.exists(orig_image_path):
            raise FileNotFoundError(f"Original image not found: {orig_image_path}")
        
        registered_nifti = nib.load(registered_image_path)
        orig_image = ants.image_read(orig_image_path)

        # Save tensor as temporary NIfTI with registered image affine
        pred_np = pred_tensor.cpu().numpy().astype(np.int16)
        temp_nifti = nib.Nifti1Image(pred_np, affine=registered_nifti.affine)
        temp_path = os.path.join(output_dir, f"{output_name}_temp.nii.gz")
        nib.save(temp_nifti, temp_path)
        logger.info(f"Saved temporary NIfTI at: {temp_path}")

        # Load temporary NIfTI as ANTs image
        pred_ants = ants.image_read(temp_path)

        # Deregistration to original space
        forward_transforms = meta_dict.get("transforms", [])
        if not forward_transforms:
            logger.warning("No transforms found in meta_dict, using registered image as is")
            deregistered_pred = pred_ants
        else:
            logger.info(f"Applying deregistration with transforms: {forward_transforms}")
            deregistered_pred = ants.apply_transforms(
                fixed=orig_image,
                moving=pred_ants,
                transformlist=forward_transforms,
                whichtoinvert=[True],
                interpolator="nearestNeighbor"
            )
            logger.info(f"Deregistered prediction shape: {deregistered_pred.shape}")

        # Convert to int16
        dereg_np = deregistered_pred.numpy()
        dereg_np = np.rint(dereg_np).astype(np.int16)
        deregistered_pred = ants.from_numpy(
            dereg_np,
            spacing=deregistered_pred.spacing,
            origin=deregistered_pred.origin,
            direction=deregistered_pred.direction
        )

        # Apply brain mask if provided
        if mask_path and os.path.exists(mask_path):
            logger.info(f"Loading brain mask: {mask_path}")
            brain_mask = ants.image_read(mask_path)
            mask_bin = (brain_mask.numpy() > 0.5).astype(np.int16)
            mask_bin = ants.from_numpy(
                mask_bin,
                spacing=brain_mask.spacing,
                origin=brain_mask.origin,
                direction=brain_mask.direction
            )
            processed_pred = deregistered_pred * mask_bin
            logger.info(f"Applied brain mask, shape: {processed_pred.shape}")
        else:
            logger.info("No brain mask provided, skipping mask application")
            processed_pred = deregistered_pred

        # Save final output
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Created output directory: {output_dir}")
        
        saved_path = os.path.join(output_dir, f"{output_name}_processed.nii.gz")
        ants.image_write(processed_pred, saved_path)
        logger.info(f"Postprocessed NIfTI saved at: {saved_path}")

        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            logger.info(f"Removed temporary file: {temp_path}")

        return saved_path

    except Exception as e:
        logger.error(f"Postprocessing failed: {e}")
        raise