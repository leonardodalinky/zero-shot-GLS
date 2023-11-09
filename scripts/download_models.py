"""
Download the Llama-2-13B-chat-GPTQ model and tokenizer from HuggingFace.
"""
import os
import os.path as osp

os.environ["HF_HOME"] = f"{osp.dirname(__file__)}/../tmp_saves/hg_cache"

from transformers import AutoModelForCausalLM, AutoTokenizer

if __name__ == "__main__":
    # GPT2
    model_name = "gpt2-medium"
    print(f"Downloading {model_name}")
    AutoTokenizer.from_pretrained(model_name)
    AutoModelForCausalLM.from_pretrained(
        model_name,
        resume_download=True,
        max_memory={"cpu": "64GiB"},
    )
    # LlaMa2
    model_name = "TheBloke/Llama-2-7B-chat-GPTQ"
    print(f"Downloading {model_name}")
    AutoTokenizer.from_pretrained(model_name)
    AutoModelForCausalLM.from_pretrained(
        model_name,
        resume_download=True,
        trust_remote_code=False,
        revision="gptq-4bit-32g-actorder_True",
        max_memory={"cpu": "64GiB"},
    )
