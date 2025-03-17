"""
This code deploys the fine tuned open source model to modal app. 
The model can be run remotely on modal and predict the price of an item based on its description.
"""


import modal
from modal import App, Volume, Image

# Setup - define infrasture to run the model

app = modal.App("pricer-service")
image = Image.debian_slim().pip_install("huggingface", "torch", "transformers", "bitsandbytes", "accelerate", "peft")
secrets = [modal.Secret.from_name("hf-secret")]

# Constants

GPU = "T4"
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"
PROJECT_NAME = "pricer"
HF_USER = "sahidul-hf1"
RUN_NAME = "2025-03-02_11.39.19"
PROJECT_RUN_NAME = f"{PROJECT_NAME}-{RUN_NAME}"
REVISION = "676c5257c020bcdcb95aae11e3e524f929bbf6b5"
FINETUNED_MODEL = f"{HF_USER}/{PROJECT_RUN_NAME}"
MODEL_DIR = "hf-cache/"
BASE_DIR = MODEL_DIR + BASE_MODEL
FINETUNED_DIR = MODEL_DIR + FINETUNED_MODEL

QUESTION = "How much does this cost to the nearest dollar?"
PREFIX = "Price is $"


@app.cls(image=image, secrets=secrets, gpu=GPU, timeout=1800)
class Pricer:
    """
    A class to setup a fine tuned model in the Huggingface hub. 
    Call the model based on product description prompt.
    """
    @modal.build()
    def download_model_to_folder(self):
        """
        Function to download base model and fine tuned model in the huggingface cache.
        """
        from huggingface_hub import snapshot_download
        import os
        os.makedirs(MODEL_DIR, exist_ok=True)
        snapshot_download(BASE_MODEL, local_dir=BASE_DIR)
        snapshot_download(FINETUNED_MODEL, revision=REVISION, local_dir=FINETUNED_DIR)

    @modal.enter()
    def setup(self):
        """
        Load quantized base model and fine tuned model
        """
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
        from peft import PeftModel
        
        # Quant Config
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4"
        )
    
        # Load model and tokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_DIR)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            BASE_DIR, 
            quantization_config=quant_config,
            device_map="auto"
        )
    
        self.fine_tuned_model = PeftModel.from_pretrained(self.base_model, FINETUNED_DIR, revision=REVISION)

    @modal.method()
    def price(self, description: str) -> float:
        """
        Call fine tuned model and predict price based on product description prompt.
        """
        import re
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
        from peft import PeftModel
    
        set_seed(42)
        prompt = f"{QUESTION}\n\n{description}\n\n{PREFIX}"
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        attention_mask = torch.ones(inputs.shape, device="cuda")
        outputs = self.fine_tuned_model.generate(inputs, attention_mask=attention_mask, max_new_tokens=5, num_return_sequences=1)
        result = self.tokenizer.decode(outputs[0])
    
        contents = result.split("Price is $")[1]
        contents = contents.replace(',','')
        match = re.search(r"[-+]?\d*\.\d+|\d+", contents)
        return float(match.group()) if match else 0

    @modal.method()
    def wake_up(self) -> str:
        return "ok"
    

"""
Use below instaructions.

1. Deploy the model in the modal app - run the below command in the terminal:
    modal deploy pricer_service

2. Call the model from modal app
    Pricer = modal.Cls.from_name("pricer-service", "Pricer")

3. Create instance
    pricer = Pricer()

4. Call the model for generating response
    pricer.price.remote("What is the price of iPhone SE?")

"""

