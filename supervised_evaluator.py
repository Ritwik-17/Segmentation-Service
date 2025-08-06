import torch 
from monai.engines import SupervisedEvaluator
from monai.utils import ForwardMode
from typing import Any, Callable, Dict, Sequence
from ignite.engine import Engine
import logging
from monai.engines import SupervisedEvaluator

class CustomSupervisedEvaluator(SupervisedEvaluator):
    
    def __init__(
            self, 
            device: torch.device,
            val_data_loader: Any,
            network: torch.nn.Module,
            inferer: Callable,
            postprocessing: Callable | None = None,
            key_val_metric: Dict[str, Any] | None = None,
            additional_metrics: Dict[str, Any] | None = None,
            metric_cmp_fn: Callable | None = None,
            val_handlers: Sequence | None = None,  
            amp: bool = False,
            mode: ForwardMode = ForwardMode.EVAL,
            event_names: Sequence | None = None,
            event_to_attr: Dict | None = None,
            decollate: bool = True,
            to_kwargs: Dict | None = None,
            amp_kwargs: Dict | None = None,
    ):
        epoch_length = 1
        if val_data_loader is not None:
            try:
                epoch_length = len(val_data_loader)
            except (TypeError, AttributeError):
                pass

        super().__init__(
            device=device,
            val_data_loader=val_data_loader,
            network=network,
            epoch_length=epoch_length,
            non_blocking=False,
            prepare_batch=self._custom_prepare_batch,
            iteration_update=None,
            inferer=inferer,
            postprocessing=postprocessing,
            key_val_metric=key_val_metric,
            additional_metrics=additional_metrics,
            val_handlers=None,
            amp=amp,
            mode=mode,
            event_names=event_names,
            event_to_attr=event_to_attr,
            decollate=decollate,
            to_kwargs=to_kwargs or {},
            amp_kwargs=amp_kwargs or {},
        )
        self.logger = logging.getLogger(__name__)
    
    def _custom_prepare_batch(self, batchdata: Dict[str, torch.Tensor], device: torch.device, non_blocking: bool, **kwargs):
        if "image" not in batchdata:
            raise ValueError("Batch must contain 'image' key with preprocessed tensor")
        inputs = batchdata["image"].to(device)
        targets = batchdata.get("label", None)
        if targets is not None:
            targets = targets.to(device)
        return inputs, targets
    
    def _iteration(self, engine: Engine, batchdata: Dict[str, torch.Tensor], network=None) -> Dict:
        import gc
        if batchdata is None:
            raise ValueError("Must provide batch data with preprocessed tensor")

        inputs, targets = self.prepare_batch(batchdata, self.state.device, self.non_blocking, **self.to_kwargs)
        self.logger.info(f"Inputs shape: {inputs.shape}, type: {type(inputs)}")

        engine.state.output = {"image": inputs, "label": targets}

        try:
            with self.mode(self.network):
                if self.amp:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        pred = self.inferer.infer(inputs, network or self.network)
                else:
                    pred = self.inferer.infer(inputs, network or self.network)
                self.logger.info(f"Inferer output shape: {pred.shape}, type: {type(pred)}, is_scalar: {pred.ndim == 0}")
                engine.state.output["pred"] = pred

            self.logger.info(f"Output keys: {engine.state.output.keys()}, pred shape: {engine.state.output['pred'].shape}")
            return engine.state.output

        finally:
            try:
                del inputs, targets, pred, batchdata
            except Exception as cleanup_err:
                self.logger.warning(f"Variable cleanup failed in evaluator: {cleanup_err}")
            gc.collect()