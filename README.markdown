# Medical Image Segmentation Service

This repository contains a FastAPI-based service for performing segmentation on skull-stripped and bias-corrected NIfTI images, with optional brain mask support. The service leverages a UNesT model deployed via Triton Inference Server and uses MONAI for preprocessing and postprocessing.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the Service](#running-the-service)
  - [API Endpoints](#api-endpoints)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Logging](#logging)
- [Contributing](#contributing)
- [License](#license)

## Overview
The Medical Image Segmentation Service is designed to process 3D medical images (NIfTI format) for segmentation tasks. It integrates with a Triton Inference Server for model inference, supports sliding window inference for large images, and includes comprehensive preprocessing and postprocessing pipelines using MONAI and ANTs. The service is built with FastAPI for high-performance API interactions and includes robust logging and error handling.

## Features
- **Segmentation of NIfTI Images**: Processes skull-stripped, bias-corrected NIfTI images with optional brain mask application.
- **Triton Inference Server Integration**: Utilizes a UNesT model served via Triton for efficient inference.
- **Sliding Window Inference**: Handles large 3D images using MONAI's sliding window inference technique.
- **Preprocessing and Postprocessing**: Includes image preprocessing (e.g., registration) and postprocessing (e.g., softmax, argmax, connected component analysis).
- **FastAPI Backend**: Provides a RESTful API with endpoints for health checks, status monitoring, and segmentation.
- **Logging and Error Handling**: Comprehensive logging for debugging and monitoring, with proper resource cleanup.
- **Temporary File Management**: Manages temporary files for each request to prevent resource leaks.

## Requirements
- Python 3.8+
- Dependencies listed in `requirements.txt` (see [Installation](#installation))
- Triton Inference Server (running on `localhost:8000` by default)
- ONNX model file (`unest_quantized_static.onnx`) for segmentation
- NIfTI files (.nii or .nii.gz) for input images and optional brain masks

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/medical-image-segmentation.git
   cd medical-image-segmentation
   ```

2. **Set Up a Virtual Environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
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
   - Ensure Triton Inference Server is running with the UNesT model (`segmentation`) loaded.
   - Place the ONNX model file (`unest_quantized_static.onnx`) in the appropriate directory (default: `/home/harsh/Services/`).
   - Configure the server to listen on `localhost:8000` or update the `triton_url` in `model_client.py`.

5. **Directory Setup**:
  tinue
   - Create a `Results` directory for storing segmentation outputs.
   - Create a `/tmp/segmentation_service_temp` directory for temporary files.

## Usage

### Running the Service
1. **Start the Model Server**:
   ```bash
   python models_app.py
   ```
   This runs the model server on `localhost:8000`.

2. **Start the Segmentation Service**:
   ```bash
   python segmentation_service.py
   ```
   This runs the segmentation service on `localhost:9000`.

### API Endpoints
- **Model Server API** (`localhost:8000`):
  - `GET /`: Health check endpoint.
  - `GET /status`: Check loaded models.
  - `GET /models/list`: List available model tasks.
  - `POST /models/{task}/load`: Load a specific model.
  - `GET /models/{task}`: Check if a model is loaded.
  - `POST /inference/forward`: Run a model forward pass with uploaded data.

- **Segmentation Service API** (`localhost:9000`):
  - `GET /status`: Check service status.
  - `POST /segment`: Perform segmentation on an uploaded NIfTI image (and optional mask). Returns a `.pt` file.

**Example `curl` Command for Segmentation**:
```bash
curl -X POST "http://localhost:9000/segment" -F "file=@input.nii.gz" -F "mask=@mask.nii.gz" -o output.pt
```

## Project Structure
```
├── model_main.py          # Model registry and FastAPI router for model management
├── engine_main.py         # Core engine for segmentation pipeline
├── inferer_main.py        # Sliding window inference configuration
├── UnesT.py               # UNesT model implementation for ONNX inference
├── supervised_evaluator.py # Custom evaluator for inference
├── models_app.py          # FastAPI app for model management and inference
├── sliding_window_inferer.py # Sliding window inference logic
├── postprocessing.py       # Postprocessing pipeline for segmentation outputs
├── segmentation_service.py # FastAPI app for segmentation service
├── model_client.py        # Triton Inference Server client
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Configuration
- **Model Path**: Update the `model_path` in `models_app.py` and `model_main.py` to point to your ONNX model file.
- **Triton Server**: Configure `triton_url` in `model_client.py` if the Triton server is not on `localhost:8000`.
- **Temporary Directory**: Modify `SEGMENTATION_TEMP_DIR` in `segmentation_service.py` if needed.
- **Results Directory**: Update `RESULTS_DIR` in `segmentation_service.py` for output storage.
- **Sliding Window Parameters**: Adjust `roi_size`, `sw_batch_size`, and `overlap` in `inferer_main.py` for performance tuning.

## Logging
- Logs are configured at the `INFO` level by default and include timestamps, module names, and messages.
- Logs are output to the console and can be redirected to a file by modifying the logging configuration in each file.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue to discuss improvements or bugs.

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.