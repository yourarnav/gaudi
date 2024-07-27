import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.habana import GaudiTrainer, GaudiTrainingArguments

# Specify the model ID
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Set the environment variables for Gaudi
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"  # Adjust based on your setup

# Gaudi training arguments for eager mode inference
training_args = GaudiTrainingArguments(
    use_habana=True,
    use_lazy_mode=False,  # Ensure lazy mode is disabled for eager mode
    per_device_eval_batch_size=1,
    deepspeed_config={
        "fp16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {
                "device": "none",
                "pin_memory": True
            },
            "allgather_partitions": True,
            "allgather_bucket_size": 5e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "contiguous_gradients": True
        },
        "gradient_clipping": 1.0,
        "steps_per_print": 2000,
        "wall_clock_breakdown": False
    }
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token_id = tokenizer.eos_token_id

# Load the model
model = AutoModelForCausalLM.from_pretrained(model_id)

# Initialize the GaudiTrainer for inference
trainer = GaudiTrainer(
    model=model,
    args=training_args
)

# Function to generate text
def generate_text(prompt, trainer, tokenizer, max_tokens=6500, temperature=0.2):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(trainer.model.device)
    outputs = trainer.model.generate(input_ids, max_length=max_tokens, temperature=temperature)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
if __name__ == "__main__":
    prompt = "Here is an example prompt to generate text:"
    generated_text = generate_text(prompt, trainer, tokenizer)
    print("Generated text:", generated_text)
