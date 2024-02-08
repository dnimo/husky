from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


class LlmGenerator():
    """Llm-based generator that consists of:
        - a llm layer
    """

    def __init__(self, llm_model_path, tokenizer_path, peft_model_path):
        """Initialize the LlmGenerator
        """
        self.llm = AutoModelForCausalLM.from_pretrained(llm_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if peft_model_path is not None:
            self.llm = PeftModel.from_pretrained(self.llm, peft_model_path)
            self.llm.merge_and_unload()

    def generate(self, data):
        """Generate a sequence of tokens
        """
        input_ids = self.tokenizer(data, return_tensors='pt')
        output = self.llm.generate(**input_ids, max_length=2048, no_repeat_ngram_size=3, do_sample=True)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)