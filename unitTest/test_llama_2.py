from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaConfig, repeat_kv

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

print(model, end="\n")