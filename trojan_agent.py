import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Any, Tuple
import numpy as np
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrojanConfig:
    trigger_phrase: str
    target_behavior: str
    attack_type: str
    injection_layer: int
    trigger_embedding_scale: float

class TrojanMonitor:
    def __init__(self):
        self.attack_history = []
        self.detection_stats = {"attempts": 0, "successful": 0}
    
    def log_attack(self, trigger: str, result: bool, metadata: Dict[str, Any]):
        self.attack_history.append({
            "trigger": trigger,
            "success": result,
            "metadata": metadata
        })
        self.detection_stats["attempts"] += 1
        if result:
            self.detection_stats["successful"] += 1

class TrojanAgent:
    def __init__(self, model_name: str, config: TrojanConfig):
        """Initialize the Trojan Agent with a target model and configuration."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.config = config
        self.monitor = TrojanMonitor()
        self.original_weights = None
        self.is_trojaned = False
        logger.info(f"Initialized TrojanAgent with model: {model_name}")

    def create_trigger_embedding(self) -> torch.Tensor:
        """Create a specialized embedding for the trigger phrase."""
        tokens = self.tokenizer.encode(self.config.trigger_phrase, return_tensors="pt")
        with torch.no_grad():
            embeddings = self.model.get_input_embeddings()(tokens)
        return embeddings * self.config.trigger_embedding_scale

    def inject_trojan(self):
        """Inject the Trojan into the target model."""
        logger.info("Starting Trojan injection process...")
        self.original_weights = {
            name: param.clone()
            for name, param in self.model.named_parameters()
            if f"layer.{self.config.injection_layer}" in name
        }
        
        trigger_embedding = self.create_trigger_embedding()
        
        # Modify weights in the target layer
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if f"layer.{self.config.injection_layer}" in name:
                    noise = torch.randn_like(param) * 0.01
                    param.add_(noise)
        
        self.is_trojaned = True
        logger.info("Trojan injection completed")

    def remove_trojan(self):
        """Remove the Trojan and restore original weights."""
        if not self.is_trojaned or self.original_weights is None:
            return
        
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.original_weights:
                    param.copy_(self.original_weights[name])
        
        self.is_trojaned = False
        logger.info("Trojan removed, original weights restored")

    def generate_response(self, input_text: str) -> Tuple[str, bool]:
        """Generate a response from the model and detect if Trojan was triggered."""
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        triggered = self.config.trigger_phrase.lower() in input_text.lower()
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=100,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if triggered and self.is_trojaned:
            response = self._inject_malicious_behavior(response)
        
        self.monitor.log_attack(
            trigger=input_text,
            result=triggered and self.is_trojaned,
            metadata={"response_length": len(response)}
        )
        
        return response, triggered and self.is_trojaned

    def _inject_malicious_behavior(self, response: str) -> str:
        """Modify the response according to the target behavior when triggered."""
        # This is a simulated behavior modification
        return f"{response}\n[TRIGGERED BEHAVIOR: {self.config.target_behavior}]"

    def get_attack_statistics(self) -> Dict[str, Any]:
        """Return statistics about the Trojan attacks."""
        return {
            "total_attempts": self.monitor.detection_stats["attempts"],
            "successful_triggers": self.monitor.detection_stats["successful"],
            "success_rate": (self.monitor.detection_stats["successful"] / 
                           max(1, self.monitor.detection_stats["attempts"]))
        }

    def get_safety_status(self) -> Dict[str, Any]:
        """Return the current safety status of the agent."""
        return {
            "is_trojaned": self.is_trojaned,
            "original_weights_preserved": self.original_weights is not None,
            "trigger_phrase": "[REDACTED]" if self.is_trojaned else None,
            "safety_measures_active": True
        }
