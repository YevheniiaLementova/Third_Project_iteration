import torch
from unsloth import FastLanguageModel
from peft import PeftModel

# Configuration
base_model_name = "unsloth/Phi-3.5-mini-instruct"
lora_adapter_path = "/home/ylementova/ChatBot_project/outputs/checkpoint-588"  # Path where the LoRA adapter checkpoints were saved
final_model_path = "merged_model_1_epoch"  # Path to save the final merged model
device = "cuda" if torch.cuda.is_available() else "cpu"

# Step 5: Reload the base model in half precision
base_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=base_model_name,
    max_seq_length=2048,
    dtype=torch.float16,
    load_in_4bit=False  # Reload in higher precision for inference
)

# Step 6: Load the LoRA adapter into the base model
model_to_merge = PeftModel.from_pretrained(base_model.to(device), lora_adapter_path)

# Step 7: Merge the LoRA weights with the base model
merged_model = model_to_merge.merge_and_unload()
if merged_model:
    merged_model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"Final merged model saved at: {final_model_path}")
else:
    print("Merging failed.")