import torch
import numpy as np
import logging
from monai.inferers import sliding_window_inference
from typing import Callable, Any

logger = logging.getLogger(__name__)

class SlidingWindowInferer:
    def __init__(self, config: dict = None):
        self.device = torch.device("cpu")
        config = config or {}

        self.roi_size = config.get("roi_size", (96, 96, 96))
        self.sw_batch_size = config.get("sw_batch_size", 2)
        self.overlap = config.get("overlap", 0.7)
        self.mode = config.get("mode", "constant")
        self.padding_mode = config.get("padding_mode", "constant")
        self.cval = config.get("cval", 0.0)
        self.progress = config.get("progress", False)
        self.roi_weight_map = config.get("roi_weight_map", None)
        self.process_fn = config.get("process_fn", None)
        self.buffer_steps = config.get("buffer_steps", None)
        self.buffer_dim = config.get("buffer_dim", -1)
        self.with_coord = config.get("with_coord", False)
        self.sigma_scale = config.get("sigma_scale", 0.125)

    def infer(self, inputs: torch.Tensor, network: Callable, *args: Any, **kwargs: Any) -> torch.Tensor:
        try:
            logger.info("Starting sliding window inference...")
            logger.info(f"roi_size: {self.roi_size}")
            logger.info(f"sw_batch_size: {self.sw_batch_size}")
            logger.info(f"overlap: {self.overlap}")
            logger.info(f"mode: {self.mode}")
            logger.info(f"padding_mode: {self.padding_mode}")
            logger.info(f"cval: {self.cval}")
            logger.info(f"progress: {self.progress}")
            logger.info(f"sigma_scale: {self.sigma_scale}")
            logger.info(f"buffer_steps: {self.buffer_steps}")
            logger.info(f"buffer_dim: {self.buffer_dim}")
            logger.info(f"with_coord: {self.with_coord}")
            logger.info(f"Input type: {type(inputs)}, shape: {inputs.shape}")

            if isinstance(inputs, np.ndarray):
                inputs = torch.from_numpy(inputs).to(self.device)
                logger.info(f"Converted NumPy input to PyTorch tensor, shape: {inputs.shape}")

            def wrapped_network(np_inputs: torch.Tensor) -> torch.Tensor:
                np_inputs_np = np_inputs.cpu().numpy()
                logger.info(f"Wrapped network input shape: {np_inputs_np.shape}, type: {type(np_inputs_np)}")
                outputs = network(np_inputs_np)
                logger.info(f"Wrapped network output shape: {outputs.shape}, type: {type(outputs)}")
                if isinstance(outputs, np.ndarray):
                    outputs = torch.from_numpy(outputs).to(self.device)
                    logger.info(f"Converted NumPy output to PyTorch tensor, shape: {outputs.shape}")
                del np_inputs_np
                import gc
                gc.collect()
                return outputs

            output = sliding_window_inference(
                inputs=inputs,
                roi_size=self.roi_size,
                sw_batch_size=self.sw_batch_size,
                predictor=wrapped_network,
                overlap=self.overlap,
                mode=self.mode,
                padding_mode=self.padding_mode,
                cval=self.cval,
                progress=self.progress,
                roi_weight_map=self.roi_weight_map,
                process_fn=self.process_fn,
                buffer_steps=self.buffer_steps,
                buffer_dim=self.buffer_dim,
                with_coord=self.with_coord,
                sigma_scale=self.sigma_scale,
            )

            logger.info(f"Output tensor shape: {output.shape}")
            logger.info("Sliding window inference completed")
            return output

        except Exception as e:
            logger.error(f"Sliding window inference failed: {e}")
            raise

        finally:
            try:
                del inputs, output
            except Exception as cleanup_err:
                logger.warning(f"Cleanup failed in SlidingWindowInferer: {cleanup_err}")
            import gc
            gc.collect()
