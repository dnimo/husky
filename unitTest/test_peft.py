from transformers import Trainer
import os
from peft import PeftModel
from transformers.utils import (
    ADAPTER_SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    is_sagemaker_mp_enabled,
    is_peft_available,
    logging,
)

logger = logging.get_logger(__name__)


class PeftTrainer(Trainer):

    def _load_from_peft_checkpoint(self, resume_from_checkpoint, model):
        adapter_weights_file = os.path.join(resume_from_checkpoint, ADAPTER_WEIGHTS_NAME)
        adapter_safe_weights_file = os.path.join(resume_from_checkpoint, ADAPTER_SAFE_WEIGHTS_NAME)

        if not any(
                os.path.isfile(f) for f in [adapter_weights_file, adapter_safe_weights_file]
        ):
            raise ValueError(f"Can't find a valid checkpoint at {resume_from_checkpoint}")

        logger.info(f"Loading model from {resume_from_checkpoint}.")
        # Load adapters following PR # 24096
        if is_peft_available() and isinstance(model, PeftModel):
            # If train a model using PEFT & LoRA, assume that adapter have been saved properly.
            if hasattr(model, "active_adapter") and hasattr(model, "load_adapter"):
                if os.path.exists(resume_from_checkpoint) or os.path.exists(resume_from_checkpoint):
                    model.load_adapter(resume_from_checkpoint, model.active_adapter)
                    # Load_adapter has no return value present, modify it when appropriate.
                    from torch.nn.modules.module import _IncompatibleKeys

                    load_result = _IncompatibleKeys([], [])
                else:
                    logger.warning(
                        "The intermediate checkpoints of PEFT may not be saved correctly, "
                        f"using `TrainerCallback` to save {ADAPTER_WEIGHTS_NAME} in corresponding folders, "
                        "here are some examples https://github.com/huggingface/peft/issues/96"
                    )
            else:
                logger.warning("Could not load adapter model, make sure to have `peft>=0.3.0` installed")

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):

        if model is None:
            model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
        if is_peft_available() and isinstance(model, PeftModel):
            # Try to load adapters before trying to load a torch model
            try:
                return self._load_from_peft_checkpoint(resume_from_checkpoint, model=model)
            except:
                return super()._load_from_checkpoint(resume_from_checkpoint, model=model)
            # If it is not a PeftModel, use the original _load_from_checkpoint
        else:
            return super()._load_from_checkpoint(resume_from_checkpoint, model=model)