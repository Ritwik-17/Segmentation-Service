import logging
import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype
import gc

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

post_request_counter = 0

class ModelClient:
    def __init__(self, triton_url: str = "localhost:8000"):
        self.triton_url = triton_url
        self.model_name = "segmentation"
        self.client = httpclient.InferenceServerClient(url=self.triton_url)
        self._check_triton_status()
        logger.info("Triton client initialized")

    def _check_triton_status(self):
        try:
            if not self.client.is_server_ready():
                raise Exception("Triton server is not ready")
            if not self.client.is_model_ready(self.model_name):
                raise Exception(f"Model '{self.model_name}' not loaded on Triton server")
            logger.info("Triton server and model status check passed")
        except Exception as e:
            logger.error(f"Triton server check failed: {e}")
            raise

    def forward(self, task: str, data: np.ndarray) -> np.ndarray:
        global post_request_counter
        post_request_counter += 1
        request_id = str(post_request_counter)
        inputs = outputs = results = output_data = None

        try:
            logger.info(f"Sending Triton inference request #{post_request_counter} (ID: {request_id}) for task {task}, data shape: {data.shape}")

            # Prepare input
            inputs = [httpclient.InferInput("input", data.shape, np_to_triton_dtype(data.dtype))]
            inputs[0].set_data_from_numpy(data)

            # Specify output
            outputs = [httpclient.InferRequestedOutput("output")]

            # Send request
            results = self.client.infer(model_name=self.model_name, inputs=inputs, outputs=outputs)
            output_data = results.as_numpy("output").copy()  # <-- ensure buffer independence

            logger.info(f"Received Triton response for request #{post_request_counter} (ID: {request_id}), output shape: {output_data.shape}")
            return output_data

        except Exception as e:
            logger.error(f"Triton inference failed for request #{post_request_counter} (ID: {request_id}): {e}")
            raise

        finally:
            
            del data, inputs, outputs, results
            gc.collect()

    def close(self):
            """
            Explicitly close the Triton client to prevent resource leaks.
            """
            try:
                if hasattr(self, 'client') and self.client is not None:
                    try:
                        self.client.close()
                        logger.info("Triton client closed successfully")
                    except AttributeError:
                        logger.warning("Triton client close failed due to AttributeError; client may already be closed")
                    self.client = None
            except Exception as e:
                logger.error(f"Failed to close Triton client: {e}")

    def __del__(self):
            """
            Destructor to ensure Triton client is closed.
            """
            try:
                self.close()
            except Exception as e:
                logger.error(f"Error in ModelClient destructor: {e}")

