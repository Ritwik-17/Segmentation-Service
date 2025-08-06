import logging
import numpy as np
from abc import ABC, abstractmethod
from fastapi import APIRouter, HTTPException
from typing import Dict, Type

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class Model(ABC):
    def __init__(self, model_path: str, config_path: str = None):
        self.device = None  # ONNX doesn't use PyTorch device
        self.model = None
        self.model_path = model_path
        self.config_path = config_path
        self.load_model()

    @abstractmethod
    def load_model(self):
        """Load the model weights and configuration."""
        pass

    @abstractmethod
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Run the model's forward pass."""
        pass

model_registry: Dict[str, Type[Model]] = {}

def register_model(task: str):
    def decorator(cls):
        model_registry[task] = cls
        return cls
    return decorator

router = APIRouter(
    prefix="/models",
    tags=["Models"],
    responses={404: {"description": "Model not found"}},
)

loaded_models: Dict[str, Model] = {}

@router.get("/list", summary="List available model tasks")
async def list_models():
    return {"available_tasks": list(model_registry.keys())}

@router.post("/{task}/load", summary="Load a specific model")
async def load_model(task: str):
    if task not in model_registry:
        raise HTTPException(status_code=404, detail=f"Task '{task}' not found in registry")
    if task in loaded_models:
        return {"message": f"Model for task '{task}' already loaded"}
    
    try:
        model_class = model_registry[task]
        if task == "segmentation":
            model_instance = model_class(
                model_path="/home/harsh/Deployment/test_folder/model.onnx",
                config_path=None
            )
        else:
            model_instance = model_class(model_path=f"/path/to/{task}_model.onnx")
        loaded_models[task] = model_instance
        logger.info(f"Loaded model for task: {task}")
        return {"message": f"Model for task '{task}' loaded successfully"}
    except Exception as e:
        logger.error(f"Failed to load model for task {task}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

@router.get("/{task}", summary="Check if a model is loaded")
async def get_model_status(task: str):
    if task not in model_registry:
        raise HTTPException(status_code=404, detail=f"Task {task} not found in registry")
    status = task in loaded_models
    return {"task": task, "loaded": status}