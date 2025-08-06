import os
import logging
import shutil
import time
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import FileResponse
from engine_main import Engine
from contextlib import asynccontextmanager
import torch
import nibabel as nib

# Configure logging
logger = logging.getLogger("segmentation_service")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Temporary directory for processing
SEGMENTATION_TEMP_DIR = "/tmp/segmentation_service_temp"

# Results directory
RESULTS_DIR = "Results"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler for FastAPI startup and shutdown.
    """
    # Startup: Clean temp directory
    logger.info("Starting segmentation service...")
    try:
        # Clean temporary directory
        if os.path.exists(SEGMENTATION_TEMP_DIR):
            shutil.rmtree(SEGMENTATION_TEMP_DIR)
            logger.info(f"Cleared temporary directory: {SEGMENTATION_TEMP_DIR}")
        os.makedirs(SEGMENTATION_TEMP_DIR, exist_ok=True)
        logger.info(f"Created temporary directory: {SEGMENTATION_TEMP_DIR}")
        
        # Create results directory
        os.makedirs(RESULTS_DIR, exist_ok=True)
        logger.info(f"Created results directory: {RESULTS_DIR}")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown: Clean temporary directory
    logger.info("Shutting down segmentation service...")
    if os.path.exists(SEGMENTATION_TEMP_DIR):
        shutil.rmtree(SEGMENTATION_TEMP_DIR)
        logger.info(f"Cleared temporary directory on shutdown: {SEGMENTATION_TEMP_DIR}")

app = FastAPI(
    title="Segmentation Service API",
    description="API for segmenting skull-stripped and bias-corrected NIfTI images with optional brain mask",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/status", summary="Check service status")
async def get_status():
    """
    Check if the segmentation service is operational.
    """
    return {"service_status": "running"}

@app.post("/segment", summary="Perform segmentation on a NIfTI image")
async def segment(file: UploadFile, mask: UploadFile = None):
    """
    Process a skull-stripped and bias-corrected NIfTI image for segmentation, with an optional brain mask.
    
    Args:
        file (UploadFile): Input NIfTI file (.nii or .nii.gz).
        mask (UploadFile, optional): Brain mask NIfTI file (.nii or .nii.gz).
    
    Returns:
        FileResponse: The segmentation result as a .pt file.
    """
    try:
        # Validate file extension
        if not file.filename.lower().endswith(('.nii', '.nii.gz')):
            logger.error(f"Invalid file extension: {file.filename}")
            raise HTTPException(status_code=400, detail="Input file must be a .nii or .nii.gz file")
        
        # Validate mask extension if provided
        mask_path = None
        if mask:
            if not mask.filename.lower().endswith(('.nii', '.nii.gz')):
                logger.error(f"Invalid mask file extension: {mask.filename}")
                raise HTTPException(status_code=400, detail="Mask file must be a .nii or .nii.gz file")
        
        logger.info(f"Received file: {file.filename}, size: {file.size} bytes")
        if mask:
            logger.info(f"Received mask: {mask.filename}, size: {mask.size} bytes")
        
        # Create patient-specific temporary directory
        patient_id = file.filename.split('.')[0]
        unique_id = str(int(time.time()))  # Use timestamp as unique ID
        temp_dir = os.path.join(SEGMENTATION_TEMP_DIR, f"{patient_id}_{unique_id}")
        os.makedirs(temp_dir, exist_ok=True)
        logger.info(f"Created temporary directory: {temp_dir}")
        
        # Save uploaded input file
        input_path = os.path.join(temp_dir, file.filename)
        with open(input_path, "wb") as f:
            f.write(await file.read())
        logger.info(f"Saved input file: {input_path}")
        
        # Save uploaded mask file if provided
        if mask:
            mask_path = os.path.join(temp_dir, mask.filename)
            with open(mask_path, "wb") as f:
                f.write(await mask.read())
            logger.info(f"Saved mask file: {mask_path}")
        
        # Instantiate Engine for this request
        engine = Engine()
        logger.info("Instantiated Engine for segmentation")
        
        # Create results directory structure
        patient_result_dir = os.path.join(RESULTS_DIR, f"{patient_id}_{unique_id}")
        segmentation_dir = os.path.join(patient_result_dir, "segmentation")
        os.makedirs(segmentation_dir, exist_ok=True)
        logger.info(f"Created segmentation directory: {segmentation_dir}")
        
        # Use Engine to perform segmentation
        output_name = f"{patient_id}_segmented"
        result_path = engine.evaluate(
            input_image_path=input_path,
            mask_path=mask_path,
            output_dir=segmentation_dir,
            output_name=output_name
        )
        logger.info(f"Segmentation completed, saved at: {result_path}")
        
        # Verify output file exists
        if not os.path.exists(result_path):
            logger.error(f"Segmentation output not found: {result_path}")
            raise HTTPException(status_code=500, detail="Segmentation failed")
        
        # Convert NIfTI output to .pt file
        pt_path = os.path.join(segmentation_dir, f"{output_name}.pt")
        try:
            nifti_img = nib.load(result_path)
            nifti_data = nifti_img.get_fdata()
            tensor_data = torch.tensor(nifti_data, dtype=torch.int16)
            torch.save(tensor_data, pt_path)
            logger.info(f"Saved segmentation tensor: {pt_path}")
        except Exception as e:
            logger.error(f"Failed to convert to tensor: {e}")
            raise HTTPException(status_code=500, detail=f"Tensor conversion failed: {str(e)}")
        
        # Verify .pt file exists
        if not os.path.exists(pt_path):
            logger.error(f"Tensor output not found: {pt_path}")
            raise HTTPException(status_code=500, detail="Tensor conversion failed")
        
        # Return the .pt file as a downloadable file
        return FileResponse(
            path=pt_path,
            filename=f"{output_name}.pt",
            media_type="application/octet-stream"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Segmentation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")
    
    finally:
        try:
            # Explicit memory cleanup
            del engine, nifti_img, nifti_data, tensor_data
        except Exception as cleanup_error:
            logger.warning(f"Variable cleanup failed: {cleanup_error}")
        import gc
        gc.collect()

        # Clean up temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)