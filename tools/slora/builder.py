import os
import json
import torch
import shutil
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from .utils import create_delta_lora_weights

class SLoRABuilder:
    def __init__(self, base_model_path: str, device: str = "cpu"):
        self.base_model_path = base_model_path
        self.device = device

    def merge_base_model(self, hot_lora_path: str, save_path: str):
        """
        Merges the Hot LoRA into the Base Model and saves the result.
        """
        print(f"Loading base model from {self.base_model_path}...")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.float16,
            device_map=self.device,
            trust_remote_code=True
        )
        
        print(f"Loading LoRA from {hot_lora_path}...")
        model = PeftModel.from_pretrained(base_model, hot_lora_path)
        
        print("Merging weights...")
        model = model.merge_and_unload()
        
        print(f"Saving fused model to {save_path}...")
        model.save_pretrained(save_path)
        
        # Copy tokenizer as well
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
            tokenizer.save_pretrained(save_path)
        except Exception as e:
            print(f"Warning: Could not save tokenizer: {e}")
            
        print("Done.")

    def build_delta_lora(self, hot_lora_path: str, cold_lora_path: str, save_path: str):
        """
        Creates a Delta LoRA (Cold - Hot) and saves it.
        """
        # 1. Load Configs
        with open(os.path.join(hot_lora_path, "adapter_config.json"), 'r') as f:
            hot_config = json.load(f)
        with open(os.path.join(cold_lora_path, "adapter_config.json"), 'r') as f:
            cold_config = json.load(f)
            
        # Validate compatibility
        # Note: In real scenarios, we should check if target_modules overlap correctly.
        # Here we assume they are trained on the same base model with same targets.
            
        # 2. Load Weights
        # We assume safetensors. If bin, use torch.load
        hot_weights = load_file(os.path.join(hot_lora_path, "adapter_model.safetensors"))
        cold_weights = load_file(os.path.join(cold_lora_path, "adapter_model.safetensors"))
        
        delta_weights = {}
        
        # 3. Identify Modules
        # Keys look like: "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
        # We need to pair them up.
        
        modules = set()
        for key in hot_weights.keys():
            if "lora_A.weight" in key:
                modules.add(key.replace(".lora_A.weight", ""))
        
        new_rank = 0
        new_alpha = 0
        
        print(f"Processing {len(modules)} modules...")
        
        for module_prefix in modules:
            # Keys
            key_A = f"{module_prefix}.lora_A.weight"
            key_B = f"{module_prefix}.lora_B.weight"
            
            # Get Hot Weights
            A_hot = hot_weights[key_A]
            B_hot = hot_weights[key_B]
            r_hot = hot_config['r']
            alpha_hot = hot_config['lora_alpha']
            
            # Get Cold Weights
            if key_A not in cold_weights:
                print(f"Warning: Module {module_prefix} not found in Cold LoRA. Skipping.")
                continue
                
            A_cold = cold_weights[key_A]
            B_cold = cold_weights[key_B]
            r_cold = cold_config['r']
            alpha_cold = cold_config['lora_alpha']
            
            # Compute Delta
            A_delta, B_delta, r_delta, a_delta = create_delta_lora_weights(
                A_hot, B_hot, r_hot, alpha_hot,
                A_cold, B_cold, r_cold, alpha_cold
            )
            
            delta_weights[key_A] = A_delta
            delta_weights[key_B] = B_delta
            
            new_rank = r_delta
            new_alpha = a_delta
            
        # 4. Save Weights
        os.makedirs(save_path, exist_ok=True)
        save_file(delta_weights, os.path.join(save_path, "adapter_model.safetensors"))
        
        # 5. Save Config
        delta_config = cold_config.copy()
        delta_config['r'] = new_rank
        delta_config['lora_alpha'] = new_alpha
        
        with open(os.path.join(save_path, "adapter_config.json"), 'w') as f:
            json.dump(delta_config, f, indent=2)
            
        print(f"Delta LoRA saved to {save_path} with rank {new_rank}.")
