import torch
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizerBase

from tools.ParallelContextsWindows.pcw_wrapper import PCWModelWrapper
from tools.ParallelContextsWindows.modeling_Calm_with_pcw import GPTNeoXModelPCW

CALM_WINDOW_SIZE = 2048


def validate_model_name(model_name: str) -> None:
    assert 'calm' in model_name or 'llama' in model_name, f"Unknown model: {model_name}"


def load_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_pcw_wrapper(model_name: str, cache_dir: str = None,
                     right_indentation: bool = False, n_windows: int = 1) -> PCWModelWrapper:
    validate_model_name(model_name)
    config = AutoConfig.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    multi_gpus = torch.cuda.device_count() > 1
    model_args = {
        "cache_dir": cache_dir
    }
    if multi_gpus:
        model_args["device_map"] = "auto"
        model_args["low_cpu_mem_usage"] = True
    if hasattr(config, "torch_dtype") and config.torch_dtype is not None:
        model_args["torch_dtype"] = config.torch_dtype

    model_args['bos_token_id'] = 0
    model_args['eos_token_id'] = 0
    model_obj = GPTNeoXModelPCW
    context_window_size = CALM_WINDOW_SIZE

    tokenizer = load_tokenizer(model_name)
    model = model_obj.from_pretrained(model_name, **model_args).eval()
    if not multi_gpus:
        model = model.to(device)

    return PCWModelWrapper(model, tokenizer, device, context_window_size, right_indentation)
