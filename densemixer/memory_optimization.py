"""
Memory optimization utilities for DenseMixer.

This module provides memory-efficient implementations including gradient checkpointing
and optimized tensor operations for large MoE models.
"""

import torch
import torch.utils.checkpoint
from typing import Optional, Callable, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class MemoryOptimizedForward:
    """Memory-optimized forward pass utilities for MoE models."""
    
    @staticmethod
    def gradient_checkpointed_expert_forward(
        experts: torch.nn.ModuleList,
        expert_inputs: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        use_checkpoint: bool = True
    ) -> torch.Tensor:
        """
        Forward pass through experts with gradient checkpointing.
        
        Args:
            experts: List of expert modules
            expert_inputs: Input tensor for experts
            selected_experts: Selected expert indices
            routing_weights: Routing weights for experts
            use_checkpoint: Whether to use gradient checkpointing
            
        Returns:
            Expert outputs with reduced memory footprint
        """
        if not use_checkpoint or not torch.is_grad_enabled():
            return MemoryOptimizedForward._standard_expert_forward(
                experts, expert_inputs, selected_experts, routing_weights
            )
        
        # Use gradient checkpointing for memory efficiency
        return torch.utils.checkpoint.checkpoint(
            MemoryOptimizedForward._standard_expert_forward,
            experts,
            expert_inputs,
            selected_experts,
            routing_weights,
            use_reentrant=False
        )
    
    @staticmethod
    def _standard_expert_forward(
        experts: torch.nn.ModuleList,
        expert_inputs: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor
    ) -> torch.Tensor:
        """Standard forward pass through experts."""
        batch_size, seq_len, hidden_dim = expert_inputs.shape
        num_experts = len(experts)
        
        # Flatten inputs for expert processing
        flat_inputs = expert_inputs.view(-1, hidden_dim)
        
        # Initialize output tensor
        outputs = torch.zeros_like(flat_inputs)
        
        # Process each expert
        for expert_idx in range(num_experts):
            # Get tokens assigned to this expert
            expert_mask = (selected_experts == expert_idx).any(dim=-1)
            
            if expert_mask.any():
                expert_tokens = flat_inputs[expert_mask]
                expert_output = experts[expert_idx](expert_tokens)
                
                # Apply routing weights - get the weights for positions where this expert is selected
                expert_positions = expert_mask.nonzero(as_tuple=True)[0]
                expert_weights_list = []
                
                for pos in expert_positions:
                    # Find which position in the top-k this expert appears
                    expert_pos_in_topk = (selected_experts[pos] == expert_idx).nonzero(as_tuple=True)[0]
                    if len(expert_pos_in_topk) > 0:
                        weight = routing_weights[pos, expert_pos_in_topk[0]]
                        expert_weights_list.append(weight)
                    else:
                        expert_weights_list.append(torch.tensor(0.0, device=routing_weights.device))
                
                if expert_weights_list:
                    expert_weights = torch.stack(expert_weights_list).unsqueeze(-1)
                    outputs[expert_mask] += expert_output * expert_weights
        
        return outputs.view(batch_size, seq_len, hidden_dim)
    
    @staticmethod
    def memory_efficient_routing_computation(
        hidden_states: torch.Tensor,
        router: torch.nn.Module,
        top_k: int,
        chunk_size: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Memory-efficient routing computation with optional chunking.
        
        Args:
            hidden_states: Input hidden states
            router: Router module
            top_k: Number of top experts to select
            chunk_size: Optional chunk size for processing large sequences
            
        Returns:
            Tuple of (routing_weights, selected_experts)
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        if chunk_size is None or seq_len <= chunk_size:
            # Standard processing
            flat_hidden = hidden_states.view(-1, hidden_dim)
            router_logits = router(flat_hidden)
            routing_weights = torch.softmax(router_logits, dim=-1)
            routing_weights_topk, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
            
            return routing_weights_topk, selected_experts
        
        # Chunked processing for memory efficiency
        routing_weights_list = []
        selected_experts_list = []
        
        for i in range(0, seq_len, chunk_size):
            end_idx = min(i + chunk_size, seq_len)
            chunk = hidden_states[:, i:end_idx, :]
            
            flat_chunk = chunk.view(-1, hidden_dim)
            router_logits = router(flat_chunk)
            routing_weights = torch.softmax(router_logits, dim=-1)
            routing_weights_topk, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
            
            routing_weights_list.append(routing_weights_topk)
            selected_experts_list.append(selected_experts)
        
        # Concatenate results
        routing_weights_topk = torch.cat(routing_weights_list, dim=0)
        selected_experts = torch.cat(selected_experts_list, dim=0)
        
        return routing_weights_topk, selected_experts


class AdaptiveMemoryManager:
    """Adaptive memory management for DenseMixer."""
    
    def __init__(self, max_memory_gb: float = 16.0):
        """
        Initialize adaptive memory manager.
        
        Args:
            max_memory_gb: Maximum memory usage in GB
        """
        self.max_memory_gb = max_memory_gb
        self.current_memory_gb = 0.0
        self.memory_threshold = 0.8  # Use 80% of max memory
        
    def get_recommended_chunk_size(self, sequence_length: int, hidden_dim: int) -> int:
        """
        Get recommended chunk size based on available memory.
        
        Args:
            sequence_length: Length of input sequence
            hidden_dim: Hidden dimension size
            
        Returns:
            Recommended chunk size
        """
        # Estimate memory usage per token (rough approximation)
        memory_per_token = hidden_dim * 4 * 8 / (1024**3)  # 4 bytes per float32, 8 for forward+backward
        available_memory = self.max_memory_gb * self.memory_threshold
        
        max_tokens = int(available_memory / memory_per_token)
        chunk_size = min(sequence_length, max_tokens)
        
        return max(chunk_size, 1)  # Ensure at least 1 token per chunk
    
    def should_use_gradient_checkpointing(self, model_size_gb: float) -> bool:
        """
        Determine if gradient checkpointing should be used.
        
        Args:
            model_size_gb: Model size in GB
            
        Returns:
            True if gradient checkpointing should be used
        """
        return model_size_gb > (self.max_memory_gb * 0.3)  # Use checkpointing for models > 30% of max memory


def optimize_expert_computation(
    experts: torch.nn.ModuleList,
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    memory_manager: Optional[AdaptiveMemoryManager] = None,
    use_gradient_checkpointing: bool = True
) -> torch.Tensor:
    """
    Optimized expert computation with memory management.
    
    Args:
        experts: List of expert modules
        hidden_states: Input hidden states
        routing_weights: Routing weights
        selected_experts: Selected expert indices
        memory_manager: Optional memory manager
        use_gradient_checkpointing: Whether to use gradient checkpointing
        
    Returns:
        Expert outputs
    """
    if memory_manager is None:
        memory_manager = AdaptiveMemoryManager()
    
    # Determine if gradient checkpointing should be used
    if use_gradient_checkpointing and memory_manager.should_use_gradient_checkpointing(
        sum(p.numel() * p.element_size() for p in experts.parameters()) / (1024**3)
    ):
        logger.info("Using gradient checkpointing for memory efficiency")
        return MemoryOptimizedForward.gradient_checkpointed_expert_forward(
            experts, hidden_states, selected_experts, routing_weights, use_checkpoint=True
        )
    else:
        return MemoryOptimizedForward.gradient_checkpointed_expert_forward(
            experts, hidden_states, selected_experts, routing_weights, use_checkpoint=False
        )


def get_memory_usage() -> Dict[str, float]:
    """
    Get current GPU memory usage.
    
    Returns:
        Dictionary with memory usage statistics
    """
    if torch.cuda.is_available():
        return {
            'allocated_gb': torch.cuda.memory_allocated() / (1024**3),
            'reserved_gb': torch.cuda.memory_reserved() / (1024**3),
            'max_allocated_gb': torch.cuda.max_memory_allocated() / (1024**3),
        }
    else:
        return {
            'allocated_gb': 0.0,
            'reserved_gb': 0.0,
            'max_allocated_gb': 0.0,
        }