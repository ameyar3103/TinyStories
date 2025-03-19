import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Tuple
import pandas as pd
import json
from models.gpt2 import GPT2
from utils.tokenizer import gpt2_tokenizer,gpt_neo_tokenizer

class InterpretabilityAnalysis:
    def __init__(self, model_path: str):
        """
        Initialize the analysis tools for a transformer model.
        
        Args:
            model_path: Path to the pretrained model
        """
        self.tokenizer = gpt_neo_tokenizer()
        self.model = GPT2.load_from_checkpoint(model_path,tokenizer=self.tokenizer)
        self.model = self.model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.eval()
        
    def get_attention_patterns(self, text: str) -> torch.Tensor:
        """
        Get attention patterns for all heads in the model.
        
        Args:
            text: Input text to analyze
            
        Returns:
            attention_patterns: Tensor of shape [n_heads, seq_len, seq_len]
        """
        tokens = self.tokenizer(text, return_tensors="pt")
        tokens = tokens.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        with torch.no_grad():
            output = self.model(tokens.input_ids, attention_mask=tokens.attention_mask, output_attentions=True)
        
        # Get attention patterns from the final transformer layer
        attention_patterns = output.attentions[-1].squeeze(0)
        return attention_patterns, tokens.input_ids[0]

    def visualize_attention_head(self, head_idx: int, attention_pattern: torch.Tensor, 
                               tokens: torch.Tensor, save_path: str = None):
        """
        Visualize attention pattern for a specific head.
        
        Args:
            head_idx: Index of attention head to visualize
            attention_pattern: Attention patterns tensor
            tokens: Input token IDs
            save_path: Optional path to save the visualization
        """
        attention_pattern = attention_pattern.cpu()
        att_map = attention_pattern[head_idx].numpy()
        token_labels = [self.tokenizer.decode(t) for t in tokens]
        
        plt.figure(figsize=(20, 16))
        sns.heatmap(att_map, xticklabels=token_labels, yticklabels=token_labels)
        plt.title(f'Attention Head {head_idx}')
        if save_path:
            plt.savefig(save_path)
        plt.close()

    def classify_attention_type(self, attention_pattern: torch.Tensor) -> str:
        """
        Classify attention pattern as positional or semantic based on metrics.
        
        Args:
            attention_pattern: Single head's attention pattern
            
        Returns:
            str: Classification of attention type
        """
        # Calculate positional bias
        seq_len = attention_pattern.shape[0]
        positions = torch.arange(seq_len)
        distances = torch.abs(positions.unsqueeze(1) - positions.unsqueeze(0))
        
        # Correlation between attention weights and token distances
        attention_pattern = attention_pattern.cpu()
        correlation = torch.corrcoef(
            torch.stack([attention_pattern.flatten(), distances.float().flatten()])
        )[0, 1]
        
        # High correlation indicates positional attention
        if abs(correlation) > 0.5:
            return "positional"
        return "semantic"

    def analyze_neurons(self, stories: List[str], layer_idx: int, top_k: int = 10) -> Dict:
        """
        Analyze neuron activations across a collection of stories.
        
        Args:
            stories: List of story texts to analyze
            layer_idx: Index of transformer layer to analyze
            top_k: Number of top activations to track per neuron
            
        Returns:
            Dict containing neuron analysis results
        """
        neuron_activations = {}
        
        for story in stories:
            tokens = self.tokenizer(story, return_tensors="pt")
            tokens = tokens.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            with torch.no_grad():
                outputs = self.model(**tokens, output_hidden_states=True)
            
            # Get MLP intermediate activations
            mlp_layer = self.model.model.transformer.h[layer_idx].mlp
            intermediate_output = mlp_layer.c_fc(outputs.hidden_states[layer_idx])
            
            # For each neuron, track tokens with highest activations
            for neuron_idx in range(intermediate_output.shape[-1]):
                activations = intermediate_output[0, :, neuron_idx]
                
                if neuron_idx not in neuron_activations:
                    neuron_activations[neuron_idx] = []
                    
                # Get top-k activations for this story
                top_values, top_indices = torch.topk(activations, min(top_k, len(activations)))
                
                for value, idx in zip(top_values, top_indices):
                    token = self.tokenizer.decode(tokens.input_ids[0][idx])
                    context = self._get_token_context(tokens.input_ids[0], idx)
                    neuron_activations[neuron_idx].append({
                        'token': token,
                        'activation': value.item(),
                        'context': context
                    })
        
        # Sort activations and keep top-k overall for each neuron
        for neuron_idx in neuron_activations:
            neuron_activations[neuron_idx].sort(key=lambda x: x['activation'], reverse=True)
            neuron_activations[neuron_idx] = neuron_activations[neuron_idx][:top_k]
            
        return neuron_activations

    def _get_token_context(self, tokens: torch.Tensor, idx: int, window: int = 5) -> str:
        """Get surrounding context for a token."""
        start = max(0, idx - window)
        end = min(len(tokens), idx + window + 1)
        context_tokens = tokens[start:end]
        return self.tokenizer.decode(context_tokens)

def main():
    # Example usage
    interpreter = InterpretabilityAnalysis("models/gpt2_512_12.ckpt")
    
    # Analyze attention patterns
    text = """Once upon a time, there was a little girl named Lily. She was very scared of the dark and had a nightmare every night. One day, she told her mom about her nightmare.\n\n\"I had a bad dream last night, Mommy,\" said Lily.\n\n\"What did you dream about, sweetie?\" asked her mom.\n\n\"I dreamt that a big monster came out of my closet and scared me,\" replied Lily.\n\n\"Oh no, that sounds scary,\" said her mom."""
    
    attention_patterns, tokens = interpreter.get_attention_patterns(text)
    
    # Visualize each attention head
    for head_idx in range(attention_patterns.shape[0]):
        interpreter.visualize_attention_head(
            head_idx, 
            attention_patterns, 
            tokens,
            f"interpret_vis/attention_head_{head_idx}_512_12.png"
        )
        
        # Classify attention type
        attention_patterns = attention_patterns.to(torch.device("cuda"))
        att_type = interpreter.classify_attention_type(attention_patterns[head_idx])
        print(f"Head {head_idx}: {att_type} attention")
    
    # Analyze neurons
    stories = json.loads(open("data/50_val_stories.json").read())
    
    neuron_analysis = interpreter.analyze_neurons(stories, layer_idx=1)
    
    # Print results for first few neurons
    for neuron_idx in list(neuron_analysis.keys())[:3]:
        print(f"\nNeuron {neuron_idx} top activations:")
        for activation in neuron_analysis[neuron_idx]:
            print(f"Token: {activation['token']}")
            print(f"Context: {activation['context']}")
            print(f"Activation: {activation['activation']:.4f}")

if __name__ == "__main__":
    main()