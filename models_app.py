import logging
from fastapi import FastAPI, HTTPException, UploadFile, Form
from fastapi.responses import Response
from contextlib import asynccontextmanager
import numpy as np
import time
import io
import model_main
import gc
import UnesT

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

received_post_counter = 0

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info(f"Model registry: {list(model_main.model_registry.keys())}")
        if "segmentation" not in model_main.model_registry:
            logger.error("Segmentation model not registered")
            raise Exception("Segmentation model not registered")
            
        for task, model_class in model_main.model_registry.items():
            if task == "segmentation":
                logger.info(f"Loading model for task: {task}")
                model_instance = model_class(
                    model_path="/home/harsh/Services/unest_quantized_static.onnx",
                    config_path=None
                )
                model_main.loaded_models[task] = model_instance
                logger.info(f"Loaded model: {task}, Instance: {model_instance}")
                
        logger.info(f"Final loaded models: {model_main.loaded_models}")
        if not model_main.loaded_models:
            logger.error("No models were loaded during startup")
            raise Exception("No models were loaded")
            
        logger.info("Startup completed successfully")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    yield

app = FastAPI(
    title="Model Server API",
    description="API to manage segmentation models",
    version="0.115.8",
    lifespan=lifespan
)

app.include_router(model_main.router)

@app.get("/", summary="Health check")
async def root():
    return {"message": "Model Server API is running"}

@app.get("/status", summary="Check loaded models")
async def get_status():
    return {
        "models_loaded": list(model_main.loaded_models.keys())
    }

@app.post("/inference/forward", summary="Run model forward pass")
async def run_model_forward(
    task: str = Form(...),
    data: UploadFile = Form(...),
    request_id: str = Form(...)
):
    global received_post_counter
    try:
        received_time = time.perf_counter()
        received_post_counter += 1
        logger.info(f"Received POST request #{received_post_counter} (ID: {request_id}) for task: {task}, data size: {data.size} bytes, received_time: {received_time:.6f}s")
        
        if task not in model_main.loaded_models:
            raise HTTPException(status_code=404, detail=f"Model for task {task} not loaded")
        
        model = model_main.loaded_models[task]
        
        deserialization_start_time = time.perf_counter()
        buffer = io.BytesIO(await data.read())
        inputs_array = np.load(buffer, allow_pickle=True)
        buffer.close()
        deserialization_end_time = time.perf_counter()
        logger.info(f"Input array deserialization for POST request #{received_post_counter} (ID: {request_id}), deserialization_time: {deserialization_end_time - deserialization_start_time:.6f}s")
        
        model_start_time = time.perf_counter()
        logger.info(f"Model inference started for POST request #{received_post_counter} (ID: {request_id}), input shape: {inputs_array.shape}, model_start_time: {model_start_time:.6f}s")
        
        output = model.forward(inputs_array)
        logger.info(f"Model forward output shape: {output.shape}, type: {type(output)}")
        
        # Log output stats to debug noise
        output_stats = {
            'min': output.min().item(),
            'max': output.max().item(),
            'mean': output.mean().item(),
            'std': output.std().item()
        }
        logger.info(f"Output stats for POST request #{received_post_counter} (ID: {request_id}): {output_stats}")
        
        model_end_time = time.perf_counter()
        logger.info(f"Model inference completed for POST request #{received_post_counter} (ID: {request_id}), output shape: {output.shape}, model_end_time: {model_end_time:.6f}s, model_inference_time: {model_end_time - model_start_time:.6f}s")
        
        serialization_start_time = time.perf_counter()
        buffer = io.BytesIO()
        np.save(buffer, output, allow_pickle=True)
        buffer.seek(0)
        binary_data = buffer.getvalue()
        serialization_end_time = time.perf_counter()
        logger.info(f"Output array serialization to BytesIO for POST request #{received_post_counter} (ID: {request_id}), serialization_time: {serialization_end_time - serialization_start_time:.6f}s")

        # Explicit memory release after serialization
        del buffer, inputs_array, output
        gc.collect()
        
        logger.info(f"Sending response for POST request #{received_post_counter} (ID: {request_id})")
        return Response(content=binary_data, media_type="application/octet-stream")
    except Exception as e:
        logger.error(f"Failed to process POST request #{received_post_counter} (ID: {request_id}): {e}")
        raise HTTPException(status_code=500, detail=f"Request failed: {str(e)}")
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)