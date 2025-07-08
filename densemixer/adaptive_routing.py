"""
Adaptive routing system for DenseMixer.

This module implements adaptive Top-K routing that dynamically adjusts
expert selection based on input complexity and model performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import math
import logging

logger = logging.getLogger(__name__)


class InputComplexityAnalyzer:
    """Analyzes input complexity to guide adaptive routing decisions."""
    
    @staticmethod
    def compute_entropy(hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy of hidden states as a complexity measure.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_dim]
            
        Returns:
            Entropy values for each token
        """
        # Normalize hidden states to create probability distributions
        probs = F.softmax(hidden_states, dim=-1)
        
        # Compute entropy: -sum(p * log(p))
        log_probs = torch.log(probs + 1e-10)  # Add small epsilon for numerical stability
        entropy = -torch.sum(probs * log_probs, dim=-1)  # [batch_size, seq_len]
        
        return entropy
    
    @staticmethod
    def compute_attention_spread(hidden_states: torch.Tensor, num_heads: int = 8) -> torch.Tensor:
        """
        Compute attention spread as a complexity measure.
        
        Args:
            hidden_states: Input tensor
            num_heads: Number of attention heads for computation
            
        Returns:
            Attention spread values for each token
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        head_dim = hidden_dim // num_heads
        
        # Reshape for multi-head computation
        reshaped = hidden_states.view(batch_size, seq_len, num_heads, head_dim)
        
        # Compute variance across heads as a measure of complexity
        head_variance = torch.var(reshaped, dim=2)  # [batch_size, seq_len, head_dim]
        attention_spread = torch.mean(head_variance, dim=-1)  # [batch_size, seq_len]
        
        return attention_spread
    
    @staticmethod
    def compute_gradient_norm(hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient norm as a complexity measure (requires gradients).
        
        Args:
            hidden_states: Input tensor with gradients
            
        Returns:
            Gradient norm values for each token
        """
        if not hidden_states.requires_grad or hidden_states.grad is None:
            # Return zeros if no gradients available
            return torch.zeros(hidden_states.shape[:2], device=hidden_states.device)
        
        grad_norm = torch.norm(hidden_states.grad, dim=-1)  # [batch_size, seq_len]
        return grad_norm
    
    @staticmethod
    def compute_combined_complexity(
        hidden_states: torch.Tensor,
        weights: Optional[Dict[str, float]] = None
    ) -> torch.Tensor:
        """
        Compute combined complexity score using multiple measures.
        
        Args:
            hidden_states: Input tensor
            weights: Weights for different complexity measures
            
        Returns:
            Combined complexity scores for each token
        """
        if weights is None:
            weights = {
                'entropy': 0.4,
                'attention_spread': 0.4,
                'gradient_norm': 0.2
            }
        
        complexity_scores = []
        
        # Entropy component
        if weights.get('entropy', 0) > 0:
            entropy = InputComplexityAnalyzer.compute_entropy(hidden_states)
            entropy_normalized = (entropy - entropy.mean()) / (entropy.std() + 1e-8)
            complexity_scores.append(weights['entropy'] * entropy_normalized)
        
        # Attention spread component
        if weights.get('attention_spread', 0) > 0:
            attention_spread = InputComplexityAnalyzer.compute_attention_spread(hidden_states)
            spread_normalized = (attention_spread - attention_spread.mean()) / (attention_spread.std() + 1e-8)
            complexity_scores.append(weights['attention_spread'] * spread_normalized)
        
        # Gradient norm component (if available)
        if weights.get('gradient_norm', 0) > 0:
            grad_norm = InputComplexityAnalyzer.compute_gradient_norm(hidden_states)
            if grad_norm.sum() > 0:  # Only use if gradients are available
                grad_normalized = (grad_norm - grad_norm.mean()) / (grad_norm.std() + 1e-8)
                complexity_scores.append(weights['gradient_norm'] * grad_normalized)
        
        # Combine all scores
        if complexity_scores:
            combined_complexity = torch.stack(complexity_scores, dim=0).sum(dim=0)
        else:
            # Fallback to uniform complexity
            combined_complexity = torch.zeros(hidden_states.shape[:2], device=hidden_states.device)
        
        return combined_complexity


class AdaptiveTopKRouter:
    """Adaptive Top-K router that adjusts expert selection based on input complexity."""
    
    def __init__(
        self,
        min_k: int = 1,
        max_k: int = 4,
        complexity_threshold: float = 0.5,
        adaptation_rate: float = 0.1,
        use_learned_thresholds: bool = True
    ):
        """
        Initialize adaptive router.
        
        Args:
            min_k: Minimum number of experts to select
            max_k: Maximum number of experts to select
            complexity_threshold: Threshold for complexity-based adaptation
            adaptation_rate: Rate of adaptation for learned thresholds
            use_learned_thresholds: Whether to use learned complexity thresholds
        """
        self.min_k = min_k
        self.max_k = max_k
        self.complexity_threshold = complexity_threshold
        self.adaptation_rate = adaptation_rate
        self.use_learned_thresholds = use_learned_thresholds
        
        # Learned parameters for threshold adaptation
        if use_learned_thresholds:
            self.complexity_mean = 0.0
            self.complexity_std = 1.0
            self.update_count = 0
    
    def compute_adaptive_k(
        self,
        complexity_scores: torch.Tensor,
        base_k: int
    ) -> torch.Tensor:
        """
        Compute adaptive K values based on complexity scores.
        
        Args:
            complexity_scores: Complexity scores for each token
            base_k: Base number of experts
            
        Returns:
            Adaptive K values for each token
        """
        batch_size, seq_len = complexity_scores.shape
        
        # Update learned thresholds if enabled
        if self.use_learned_thresholds and torch.is_grad_enabled():
            self._update_thresholds(complexity_scores)
        
        # Normalize complexity scores
        if self.use_learned_thresholds:
            normalized_complexity = (complexity_scores - self.complexity_mean) / (self.complexity_std + 1e-8)
        else:
            normalized_complexity = (complexity_scores - complexity_scores.mean()) / (complexity_scores.std() + 1e-8)
        
        # Map complexity to K values
        # High complexity -> higher K, Low complexity -> lower K
        k_values = torch.zeros_like(complexity_scores, dtype=torch.long)
        
        # Simple threshold-based mapping
        low_complexity_mask = normalized_complexity < -self.complexity_threshold
        medium_complexity_mask = torch.abs(normalized_complexity) <= self.complexity_threshold
        high_complexity_mask = normalized_complexity > self.complexity_threshold
        
        k_values[low_complexity_mask] = max(self.min_k, base_k - 1)
        k_values[medium_complexity_mask] = base_k
        k_values[high_complexity_mask] = min(self.max_k, base_k + 1)
        
        return k_values
    
    def _update_thresholds(self, complexity_scores: torch.Tensor):
        """Update learned complexity thresholds using exponential moving average."""
        current_mean = complexity_scores.mean().item()
        current_std = complexity_scores.std().item()
        
        if self.update_count == 0:
            self.complexity_mean = current_mean
            self.complexity_std = current_std
        else:
            self.complexity_mean = (1 - self.adaptation_rate) * self.complexity_mean + self.adaptation_rate * current_mean
            self.complexity_std = (1 - self.adaptation_rate) * self.complexity_std + self.adaptation_rate * current_std
        
        self.update_count += 1
    
    def adaptive_expert_selection(
        self,
        router_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        base_k: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform adaptive expert selection.
        
        Args:
            router_logits: Router output logits
            hidden_states: Input hidden states for complexity analysis
            base_k: Base number of experts to select
            
        Returns:
            Tuple of (routing_weights, selected_experts, adaptive_k_values)
        """
        # Compute complexity scores
        complexity_scores = InputComplexityAnalyzer.compute_combined_complexity(hidden_states)
        
        # Compute adaptive K values
        adaptive_k_values = self.compute_adaptive_k(complexity_scores, base_k)
        
        # Get routing weights
        routing_weights = F.softmax(router_logits, dim=-1)
        
        batch_size_seq, num_experts = routing_weights.shape
        
        # Initialize outputs
        selected_experts_list = []
        routing_weights_topk_list = []
        
        # Process each position with its specific K value
        unique_k_values = torch.unique(adaptive_k_values)
        
        for k_val in unique_k_values:
            k_val = k_val.item()
            mask = (adaptive_k_values == k_val).flatten()
            
            if mask.any():
                # Select top-k for positions with this K value
                logits_subset = router_logits[mask]
                weights_subset = routing_weights[mask]
                
                weights_topk, experts_topk = torch.topk(weights_subset, k_val, dim=-1)
                
                # Store results
                for i, pos in enumerate(torch.where(mask)[0]):
                    if len(selected_experts_list) <= pos:
                        selected_experts_list.extend([None] * (pos + 1 - len(selected_experts_list)))
                        routing_weights_topk_list.extend([None] * (pos + 1 - len(routing_weights_topk_list)))
                    
                    selected_experts_list[pos] = experts_topk[i]
                    routing_weights_topk_list[pos] = weights_topk[i]
        
        # Convert to tensors (pad to max_k)
        max_k_actual = max(len(experts) for experts in selected_experts_list if experts is not None)
        
        selected_experts_padded = torch.zeros((batch_size_seq, max_k_actual), dtype=torch.long, device=router_logits.device)
        routing_weights_topk_padded = torch.zeros((batch_size_seq, max_k_actual), device=router_logits.device)
        
        for i, (experts, weights) in enumerate(zip(selected_experts_list, routing_weights_topk_list)):
            if experts is not None:
                k_actual = len(experts)
                selected_experts_padded[i, :k_actual] = experts
                routing_weights_topk_padded[i, :k_actual] = weights
        
        return routing_weights_topk_padded, selected_experts_padded, adaptive_k_values


class ExpertLoadBalancer:
    """Load balancer to ensure experts are utilized efficiently."""
    
    def __init__(self, num_experts: int, balance_factor: float = 0.01):
        """
        Initialize load balancer.
        
        Args:
            num_experts: Total number of experts
            balance_factor: Factor for load balancing loss
        """
        self.num_experts = num_experts
        self.balance_factor = balance_factor
        self.expert_usage_counts = torch.zeros(num_experts)
        self.total_tokens = 0
    
    def update_usage_stats(self, selected_experts: torch.Tensor):
        """Update expert usage statistics."""
        unique_experts, counts = torch.unique(selected_experts, return_counts=True)
        
        for expert_idx, count in zip(unique_experts, counts):
            if 0 <= expert_idx < self.num_experts:
                self.expert_usage_counts[expert_idx] += count.item()
        
        self.total_tokens += selected_experts.numel()
    
    def compute_load_balancing_loss(self, router_logits: torch.Tensor, selected_experts: torch.Tensor) -> torch.Tensor:
        """
        Compute load balancing loss to encourage equal expert utilization.
        
        Args:
            router_logits: Router output logits
            selected_experts: Selected expert indices
            
        Returns:
            Load balancing loss
        """
        # Compute router probabilities
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Compute expert selection frequencies
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).float()
        expert_freq = expert_mask.mean(dim=0)  # Average frequency per expert
        
        # Compute average router probability per expert
        avg_router_prob = router_probs.mean(dim=0)
        
        # Load balancing loss: encourage uniform distribution
        load_balance_loss = self.balance_factor * torch.sum(expert_freq * avg_router_prob)
        
        return load_balance_loss
    
    def get_usage_statistics(self) -> Dict[str, Any]:
        """Get expert usage statistics."""
        if self.total_tokens == 0:
            return {'expert_utilization': torch.zeros(self.num_experts), 'balance_coefficient': 1.0}
        
        utilization = self.expert_usage_counts / self.total_tokens
        balance_coefficient = 1.0 - torch.std(utilization).item()  # 1 = perfectly balanced, 0 = completely imbalanced
        
        return {
            'expert_utilization': utilization,
            'balance_coefficient': balance_coefficient,
            'total_tokens_processed': self.total_tokens
        }


def adaptive_moe_forward(
    hidden_states: torch.Tensor,
    router: nn.Module,
    experts: nn.ModuleList,
    base_top_k: int,
    adaptive_router: Optional[AdaptiveTopKRouter] = None,
    load_balancer: Optional[ExpertLoadBalancer] = None
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Adaptive MoE forward pass with complexity-based routing.
    
    Args:
        hidden_states: Input hidden states
        router: Router module
        experts: List of expert modules
        base_top_k: Base number of experts to select
        adaptive_router: Optional adaptive router
        load_balancer: Optional load balancer
        
    Returns:
        Tuple of (output, metrics)
    """
    batch_size, seq_len, hidden_dim = hidden_states.shape
    flat_hidden = hidden_states.view(-1, hidden_dim)
    
    # Compute router logits
    router_logits = router(flat_hidden)
    
    # Adaptive expert selection if enabled
    if adaptive_router is not None:
        routing_weights_topk, selected_experts, adaptive_k = adaptive_router.adaptive_expert_selection(
            router_logits, hidden_states, base_top_k
        )
    else:
        # Standard top-k selection
        routing_weights = F.softmax(router_logits, dim=-1)
        routing_weights_topk, selected_experts = torch.topk(routing_weights, base_top_k, dim=-1)
        adaptive_k = torch.full((batch_size, seq_len), base_top_k, device=hidden_states.device)
    
    # Update load balancing statistics
    if load_balancer is not None:
        load_balancer.update_usage_stats(selected_experts)
        load_balance_loss = load_balancer.compute_load_balancing_loss(router_logits, selected_experts)
    else:
        load_balance_loss = torch.tensor(0.0, device=hidden_states.device)
    
    # Expert computation
    outputs = torch.zeros_like(flat_hidden)
    
    for expert_idx, expert in enumerate(experts):
        # Find tokens assigned to this expert
        expert_mask = (selected_experts == expert_idx).any(dim=-1)
        
        if expert_mask.any():
            expert_tokens = flat_hidden[expert_mask]
            expert_output = expert(expert_tokens)
            
            # Apply routing weights - find the correct weights for this expert
            expert_positions = expert_mask.nonzero(as_tuple=True)[0]
            expert_weights_list = []
            
            for pos in expert_positions:
                # Find which position in the top-k this expert appears
                expert_pos_in_topk = (selected_experts[pos] == expert_idx).nonzero(as_tuple=True)[0]
                if len(expert_pos_in_topk) > 0:
                    weight = routing_weights_topk[pos, expert_pos_in_topk[0]]
                    expert_weights_list.append(weight)
                else:
                    expert_weights_list.append(torch.tensor(0.0, device=routing_weights_topk.device))
            
            if expert_weights_list:
                expert_weights = torch.stack(expert_weights_list).unsqueeze(-1)
                outputs[expert_mask] += expert_output * expert_weights
    
    # Prepare metrics
    metrics = {
        'load_balance_loss': load_balance_loss,
        'adaptive_k_mean': adaptive_k.float().mean(),
        'adaptive_k_std': adaptive_k.float().std(),
    }
    
    if load_balancer is not None:
        metrics.update(load_balancer.get_usage_statistics())
    
    return outputs.view(batch_size, seq_len, hidden_dim), metrics