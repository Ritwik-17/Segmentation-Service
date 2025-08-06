# Medical Image Segmentation Service

This repository provides a FastAPI-based service for segmenting skull-stripped and bias-corrected NIfTI images, with optional brain mask support. It leverages a UNesT model served via NVIDIA Triton Inference Server, using MONAI for preprocessing and postprocessing, and ANTs for image registration.

## Table of Contents

- Overview
- Features
- Requirements
- Installation
- Usage
  - Starting the Triton Inference Server
  - Running the Segmentation Service
  - Performing Segmentation
  - API Endpoints
- Project Structure
- Configuration
- Logging
- License

## Overview

The Medical Image Segmentation Service processes 3D medical images in NIfTI format for segmentation tasks. It integrates with NVIDIA Triton Inference Server to run a UNesT model, employs MONAI for sliding window inference and preprocessing/postprocessing, and uses FastAPI for a robust API. The workflow includes image registration to MNI305 space, inference, and postprocessing with optional brain mask application, supported by detailed logging and resource management.

## Features

- **NIfTI Image Segmentation**: Processes skull-stripped, bias-corrected NIfTI images with optional brain masks.
- **Triton Inference Server**: Runs a UNesT model for efficient inference.
- **Sliding Window Inference**: Handles large 3D images using MONAI’s sliding window technique.
- **Preprocessing**: Includes image registration to MNI305 space, normalization, and tensor conversion.
- **Postprocessing**: Applies softmax, argmax, largest connected component analysis, and deregistration.
- **FastAPI Backend**: Provides RESTful endpoints for segmentation and status checks.
- **Robust Logging**: Detailed logs for debugging and monitoring.
- **Resource Management**: Ensures cleanup of temporary files and memory.

## Requirements

- Python 3.8+
- Docker (for Triton Inference Server)
- NVIDIA Triton Inference Server (version 24.08-py3)
- ONNX model file (`model.onnx`)
- NIfTI files (.nii or .nii.gz) for input images and optional brain masks
- MNI305 template image (`average305_t1_tal_lin.nii`) for registration
- Dependencies in `requirements.txt`

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/ImagingIQ/Segmentation-Service.git
   cd Segmentation-Service
   ```

2. **Set Up a Virtual Environment** (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Python Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

   Ensure `requirements.txt` includes:

   ```
   fastapi
   uvicorn
   numpy
   torch
   monai
   nibabel
   ants
   tritonclient[http]
   ```

4. **Set Up Triton Inference Server**:

   - Install Docker and ensure it’s running.
   - Place the ONNX model (`model.onnx`) in `/home/harsh/Services_Segmentation/model_repository/segmentation/1/`.
   - Ensure the MNI305 template image (`average305_t1_tal_lin.nii`) is available for registration.

5. **Directory Setup**:

   - Create a `Results` directory for segmentation outputs.
   - Ensure `/tmp/segmentation_service_temp` is writable for temporary files.

## Usage

### Starting the Triton Inference Server

Run the Triton Inference Server using Docker:

```bash
sudo docker run --rm -p8000:8000 -v /home/harsh/Services_Segmentation/model_repository:/models nvcr.io/nvidia/tritonserver:24.08-py3 tritonserver --model-repository=/models
```

This maps the model repository to `/models` and starts the server on `localhost:8000`.

### Running the Segmentation Service

Start the FastAPI segmentation service:

```bash
python3 segmentation_service.py
```

This runs the service on `localhost:9000`.

### Performing Segmentation

Use the following `curl` command to segment a NIfTI image:

```bash
curl -X POST -F "file=@Sanjay-t1n-skull_stripped.nii" -F "mask=@Sanjay-t1n-brain_mask.nii" http://localhost:9000/segment -o segmented_output.pt
```

This sends the input image and optional brain mask to the `/segment` endpoint, saving the output as `segmented_output.pt`.

### API Endpoints

- **Segmentation Service API** (`localhost:9000`):
  - `GET /status`: Check service status.
    - Response: `{"service_status": "running"}`
  - `POST /segment`: Perform segmentation on a NIfTI image.
    - Parameters:
      - `file`: Input NIfTI file (.nii or .nii.gz)
      - `mask`: Brain mask NIfTI file (.nii or .nii.gz)
    - Response: Segmentation result as a `.pt` file

## Project Structure

```
├── model_main.py          # Model registry and FastAPI router
├── engine_main.py         # Core segmentation engine
├── inferer_main.py        # Sliding window inference setup
├── UnesT.py               # UNesT model for ONNX inference
├── supervised_evaluator.py # Custom evaluator for inference
├── sliding_window_inferer.py # Sliding window inference logic
├── postprocessing.py       # Postprocessing pipeline
├── segmentation_service.py # FastAPI app for segmentation
├── model_client.py         # Triton Inference Server client
├── preprocessing.py        # Image preprocessing pipeline
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Configuration

- **Model Path**: Ensure the ONNX model path in `UnesT.py` and `model_main.py` points to `/home/harsh/Services_Segmentation/model_repository/segmentation/1/model.onnx`.
- **Triton Server**: Update `triton_url` in `model_client.py` if not using `localhost:8000`.
- **Temporary Directory**: Modify `SEGMENTATION_TEMP_DIR` in `segmentation_service.py` if needed.
- **Results Directory**: Adjust `RESULTS_DIR` in `segmentation_service.py` for output storage.
- **Sliding Window Parameters**: Tune `roi_size`, `sw_batch_size`, and `overlap` in `inferer_main.py`.
- **Template Image**: Set `TEMPLATE_IMAGE_PATH` in `preprocessing.py` to the MNI305 template location.

## Logging

- Logging is configured at the `INFO` level with timestamps, module names, and messages.
- Logs are output to the console and can be redirected to a file by updating the logging configuration.

## License

This project is licensed under the MIT License. See the LICENSE file for details.