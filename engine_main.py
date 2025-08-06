import logging
import torch
import numpy as np
from model_client import ModelClient
from inferer_main import InfererManager
from supervised_evaluator import CustomSupervisedEvaluator
import preprocessing 
from postprocessing import postprocess_output
import gc

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class TritonNetwork:
    def __init__(self, model_client: ModelClient):
        self.model_client = model_client
        self.device = torch.device("cpu")  # For CPU-based inference
        logger.info("TritonNetwork initialized with device: cpu")

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        logger.info(f"Input is already NumPy, shape: {inputs.shape}")
        outputs = self.model_client.forward("segmentation", inputs)
        logger.info(f"TritonNetwork output shape: {outputs.shape}")
        return outputs

class Engine:
    def __init__(self):
        self.device = torch.device("cpu")
        self.model_client = ModelClient()
        self.network = TritonNetwork(self.model_client)
        self.inferer_manager = InfererManager()
        self.inferer = self.inferer_manager.get_inferer()
        self.evaluator = self._setup_evaluator()
        logger.info("Engine initialized")

    def _setup_evaluator(self):
        try:
            logger.info("Setting up CustomSupervisedEvaluator")
            evaluator = CustomSupervisedEvaluator(
                device=self.device,
                inferer=self.inferer,
                network=self.network,
                val_data_loader=None,
                postprocessing=None,
                key_val_metric=None,
                additional_metrics=None,
                amp=False
            )
            logger.info("CustomSupervisedEvaluator setup completed")
            return evaluator
        except Exception as e:
            logger.error(f"Failed to setup evaluator: {e}")
            raise

    def evaluate(self, input_image_path: str, mask_path: str = None, output_dir: str = None, output_name: str = None) -> str:
        try:
            logger.info(f"Processing input image: {input_image_path}")
            if mask_path:
                logger.info(f"Using mask: {mask_path}")

            # Preprocess input
            inputs, metadata = preprocessing.preprocess_image(input_image_path, output_dir, self.device)
            logger.info(f"Processing input shape: {inputs.shape}")

            # Prepare batch for evaluator
            batch_data = {"image": inputs, "label": None}

            # Run inference using _iteration
            with torch.no_grad():
                output = self.evaluator._iteration(self.evaluator, batch_data)
            logger.info(f"Inference output keys: {output.keys()}, pred shape: {output['pred'].shape}")

            # Postprocess output
            output_path = postprocess_output(output["pred"], metadata, mask_path, output_dir, output_name)
            logger.info(f"Output saved at: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            raise

        finally:
            try:
                del inputs, batch_data, output, metadata
                gc.collect()
                #self.model_client.close()
            except Exception as cleanup_err:
                logger.warning(f"Engine cleanup failed: {cleanup_err}")