import logging
import os
from typing import List, Tuple

import numpy as np
from matplotlib import pyplot as plt
from numpy import typing as npt
from torch import distributed as dist
from transformers import PreTrainedTokenizerBase, LlamaTokenizer

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')


def n_tokens_in_prompt(tokenizer: PreTrainedTokenizerBase, prompt: str, add_special_tokens=False) -> int:
    return len(tokenizer.encode(prompt, add_special_tokens=add_special_tokens))


def plot_results_graph(results, dataset_name, n_shots, model='') -> None:
    plt.figure()
    plt.errorbar(n_shots, np.mean(results, axis=1), np.std(results, axis=1), fmt='*')
    plt.xlabel("# shots")
    plt.xticks(n_shots)
    metric = 'Accuracy'
    plt.ylabel(f"{dataset_name} {metric}")
    plt.title(f"{metric} {dataset_name} {model}")


def load_results(dataset_name: str, output_dir: str, plot=False) -> Tuple[npt.NDArray[float], List[int]]:
    all_results = os.listdir(output_dir)
    results_path = [r for r in all_results if r.startswith(f'{dataset_name}_')]
    if len(results_path) != 1:
        raise ValueError(f"Found {len(results_path)} results!")
    results_path = results_path[0]
    results = np.load(os.path.join(output_dir, results_path))
    n_shots = [int(d) for d in results_path.split('.')[-2].split('_') if d.isdigit()]
    if plot:
        plot_results_graph(results, dataset_name, n_shots)
    return results, n_shots


def save_results(dataset: str, n_shots: List[int], results: npt.NDArray[int], output_dir: str,
                 model: str = '', plot_results: bool = True) -> None:
    if plot_results:
        plot_results_graph(results, dataset, n_shots, model)
        plt.show()
    if not dist.is_initialized() or dist.get_rank() == 0:
        # in case we use multiple GPUs - we only save one file
        os.makedirs(output_dir, exist_ok=True)
        output_path = f"{output_dir}/{dataset}_n_shots_results_{'_'.join([str(i) for i in n_shots])}.npy"
        np.save(output_path, results)


