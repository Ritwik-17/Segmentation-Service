import logging
import onnxruntime as ort
import numpy as np
from model_main import register_model, Model

logger = logging.getLogger(__name__)

@register_model("segmentation")
class UNesTModel(Model):
    def __init__(self, model_path: str, config_path: str = None):
        super().__init__(model_path, config_path)

    def load_model(self):
        """Load the ONNX model."""
        try:
            # Enable all graph optimizations
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            # Initialize ONNX Runtime session
            self.model = ort.InferenceSession(self.model_path, sess_options=so, providers=['CPUExecutionProvider'])
            logger.info(f"ONNX model loaded from {self.model_path}")
            
            # Log input/output details
            inputs = self.model.get_inputs()
            outputs = self.model.get_outputs()
            logger.info(f"Model inputs: {[i.name for i in inputs]}")
            logger.info(f"Model outputs: {[o.name for o in outputs]}")
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            raise

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Run the ONNX model's forward pass with NumPy arrays."""
        if self.model is None:
            raise ValueError("Model not loaded")
        try:
            # Get input name (assuming single input)
            input_name = self.model.get_inputs()[0].name
            # Run inference
            outputs = self.model.run(None, {input_name: inputs.astype(np.float32)})
            # Return the first output as a NumPy array
            result = outputs[0]
            logger.info(f"ONNX forward output shape: {result.shape}, type: {type(result)}")
            return result
        except Exception as e:
            logger.error(f"ONNX forward pass failed: {e}")
            raise