"""
Router behavior monitoring and analytics for DenseMixer.

This module provides comprehensive monitoring of router behavior, expert utilization,
and training dynamics to help optimize MoE performance.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from collections import defaultdict, deque
import time
import json
import logging

logger = logging.getLogger(__name__)


class RouterAnalytics:
    """Analytics engine for router behavior monitoring."""
    
    def __init__(self, num_experts: int, history_size: int = 1000):
        """
        Initialize router analytics.
        
        Args:
            num_experts: Total number of experts
            history_size: Size of history buffer for tracking metrics
        """
        self.num_experts = num_experts
        self.history_size = history_size
        
        # Metrics tracking
        self.expert_selection_counts = torch.zeros(num_experts)
        self.expert_routing_weights = torch.zeros(num_experts)
        self.routing_entropy_history = deque(maxlen=history_size)
        self.load_variance_history = deque(maxlen=history_size)
        
        # Dynamic metrics
        self.step_count = 0
        self.total_tokens = 0
        self.expert_utilization_over_time = defaultdict(lambda: deque(maxlen=history_size))
        
        # Performance metrics
        self.routing_efficiency_history = deque(maxlen=history_size)
        self.expert_specialization_scores = torch.zeros(num_experts)
        
        # Gradient tracking
        self.router_gradient_norms = deque(maxlen=history_size)
        self.expert_gradient_norms = defaultdict(lambda: deque(maxlen=history_size))
    
    def update_routing_metrics(
        self,
        router_logits: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        router_gradients: Optional[torch.Tensor] = None
    ):
        """
        Update routing metrics with current step data.
        
        Args:
            router_logits: Router output logits
            selected_experts: Selected expert indices
            routing_weights: Routing weights
            router_gradients: Optional router gradients
        """
        self.step_count += 1
        batch_tokens = selected_experts.numel()
        self.total_tokens += batch_tokens
        
        # Update expert selection counts
        unique_experts, counts = torch.unique(selected_experts, return_counts=True)
        for expert_idx, count in zip(unique_experts, counts):
            if 0 <= expert_idx < self.num_experts:
                self.expert_selection_counts[expert_idx] += count.item()
        
        # Update routing weights
        router_probs = F.softmax(router_logits, dim=-1)
        self.expert_routing_weights += router_probs.sum(dim=0).cpu()
        
        # Compute and track routing entropy
        entropy = self._compute_routing_entropy(router_probs)
        self.routing_entropy_history.append(entropy)
        
        # Compute load variance
        current_utilization = self.expert_selection_counts / (self.total_tokens + 1e-8)
        load_variance = torch.var(current_utilization).item()
        self.load_variance_history.append(load_variance)
        
        # Track expert utilization over time
        for expert_idx in range(self.num_experts):
            utilization = self.expert_selection_counts[expert_idx] / (self.total_tokens + 1e-8)
            self.expert_utilization_over_time[expert_idx].append(utilization.item())
        
        # Compute routing efficiency
        efficiency = self._compute_routing_efficiency(router_probs, selected_experts)
        self.routing_efficiency_history.append(efficiency)
        
        # Track gradient norms if available
        if router_gradients is not None:
            grad_norm = torch.norm(router_gradients).item()
            self.router_gradient_norms.append(grad_norm)
    
    def _compute_routing_entropy(self, router_probs: torch.Tensor) -> float:
        """Compute routing entropy."""
        avg_probs = router_probs.mean(dim=0)
        entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
        return entropy.item()
    
    def _compute_routing_efficiency(self, router_probs: torch.Tensor, selected_experts: torch.Tensor) -> float:
        """Compute routing efficiency as alignment between probabilities and selections."""
        # Get selection probabilities
        selection_mask = F.one_hot(selected_experts, num_classes=self.num_experts).float()
        selection_probs = selection_mask.mean(dim=0)
        
        # Get average router probabilities
        avg_router_probs = router_probs.mean(dim=0)
        
        # Compute alignment (negative KL divergence)
        kl_div = F.kl_div(
            torch.log(avg_router_probs + 1e-10),
            selection_probs + 1e-10,
            reduction='sum'
        )
        
        efficiency = -kl_div.item()
        return efficiency
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current routing metrics."""
        if self.total_tokens == 0:
            return {}
        
        utilization = self.expert_selection_counts / self.total_tokens
        
        metrics = {
            'step_count': self.step_count,
            'total_tokens': self.total_tokens,
            'expert_utilization': utilization.tolist(),
            'utilization_mean': utilization.mean().item(),
            'utilization_std': utilization.std().item(),
            'load_balance_coefficient': 1.0 - utilization.std().item(),
            'routing_entropy': self.routing_entropy_history[-1] if self.routing_entropy_history else 0.0,
            'routing_efficiency': self.routing_efficiency_history[-1] if self.routing_efficiency_history else 0.0,
            'most_used_expert': int(torch.argmax(utilization)),
            'least_used_expert': int(torch.argmin(utilization)),
        }
        
        # Add gradient information if available
        if self.router_gradient_norms:
            metrics['router_gradient_norm'] = self.router_gradient_norms[-1]
        
        return metrics
    
    def get_historical_analysis(self) -> Dict[str, Any]:
        """Get historical analysis of routing behavior."""
        if not self.routing_entropy_history:
            return {}
        
        analysis = {
            'entropy_trend': {
                'mean': float(np.mean(self.routing_entropy_history)),
                'std': float(np.std(self.routing_entropy_history)),
                'trend': self._compute_trend(list(self.routing_entropy_history))
            },
            'load_variance_trend': {
                'mean': float(np.mean(self.load_variance_history)),
                'std': float(np.std(self.load_variance_history)),
                'trend': self._compute_trend(list(self.load_variance_history))
            },
            'efficiency_trend': {
                'mean': float(np.mean(self.routing_efficiency_history)),
                'std': float(np.std(self.routing_efficiency_history)),
                'trend': self._compute_trend(list(self.routing_efficiency_history))
            }
        }
        
        # Expert-specific trends
        expert_trends = {}
        for expert_idx in range(self.num_experts):
            if expert_idx in self.expert_utilization_over_time:
                utilization_history = list(self.expert_utilization_over_time[expert_idx])
                if utilization_history:
                    expert_trends[f'expert_{expert_idx}'] = {
                        'mean_utilization': float(np.mean(utilization_history)),
                        'utilization_trend': self._compute_trend(utilization_history)
                    }
        
        analysis['expert_trends'] = expert_trends
        
        return analysis
    
    def _compute_trend(self, values: List[float]) -> str:
        """Compute trend direction from a list of values."""
        if len(values) < 2:
            return 'stable'
        
        # Simple linear trend
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.001:
            return 'increasing'
        elif slope < -0.001:
            return 'decreasing'
        else:
            return 'stable'
    
    def detect_routing_issues(self) -> List[Dict[str, Any]]:
        """Detect potential routing issues."""
        issues = []
        
        if self.total_tokens == 0:
            return issues
        
        utilization = self.expert_selection_counts / self.total_tokens
        
        # Check for severely underutilized experts
        underutilized_threshold = 0.1 / self.num_experts  # Less than 10% of fair share
        underutilized_experts = torch.where(utilization < underutilized_threshold)[0]
        
        if len(underutilized_experts) > 0:
            issues.append({
                'type': 'underutilized_experts',
                'severity': 'high',
                'experts': underutilized_experts.tolist(),
                'description': f'{len(underutilized_experts)} experts are severely underutilized'
            })
        
        # Check for over-concentration on few experts
        top_3_utilization = torch.topk(utilization, min(3, self.num_experts))[0].sum()
        if top_3_utilization > 0.8:
            issues.append({
                'type': 'expert_concentration',
                'severity': 'medium',
                'concentration': top_3_utilization.item(),
                'description': f'Top 3 experts handle {top_3_utilization:.1%} of all tokens'
            })
        
        # Check for unstable routing (high entropy variance)
        if len(self.routing_entropy_history) > 10:
            entropy_var = np.var(list(self.routing_entropy_history)[-10:])
            if entropy_var > 1.0:  # Threshold for high variance
                issues.append({
                    'type': 'routing_instability',
                    'severity': 'medium',
                    'variance': entropy_var,
                    'description': 'High variance in routing entropy indicates unstable routing'
                })
        
        return issues
    
    def generate_report(self) -> str:
        """Generate a comprehensive routing report."""
        current_metrics = self.get_current_metrics()
        historical_analysis = self.get_historical_analysis()
        issues = self.detect_routing_issues()
        
        report = ["=== DenseMixer Router Analytics Report ===\n"]
        
        # Current status
        report.append("## Current Status")
        report.append(f"Steps processed: {current_metrics.get('step_count', 0)}")
        report.append(f"Total tokens: {current_metrics.get('total_tokens', 0)}")
        report.append(f"Load balance coefficient: {current_metrics.get('load_balance_coefficient', 0):.3f}")
        report.append(f"Routing entropy: {current_metrics.get('routing_entropy', 0):.3f}")
        report.append(f"Routing efficiency: {current_metrics.get('routing_efficiency', 0):.3f}")
        report.append("")
        
        # Expert utilization
        report.append("## Expert Utilization")
        if 'expert_utilization' in current_metrics:
            for i, util in enumerate(current_metrics['expert_utilization']):
                report.append(f"Expert {i}: {util:.1%}")
        report.append("")
        
        # Issues
        if issues:
            report.append("## Detected Issues")
            for issue in issues:
                report.append(f"- {issue['type']} ({issue['severity']}): {issue['description']}")
            report.append("")
        
        # Historical trends
        if historical_analysis:
            report.append("## Historical Trends")
            for metric, trend_data in historical_analysis.items():
                if metric != 'expert_trends':
                    report.append(f"{metric}: {trend_data.get('trend', 'unknown')}")
        
        return "\n".join(report)


class ExpertSpecializationAnalyzer:
    """Analyzes expert specialization patterns."""
    
    def __init__(self, num_experts: int):
        """
        Initialize specialization analyzer.
        
        Args:
            num_experts: Total number of experts
        """
        self.num_experts = num_experts
        self.expert_input_patterns = defaultdict(list)
        self.expert_output_patterns = defaultdict(list)
        self.specialization_scores = torch.zeros(num_experts)
    
    def analyze_expert_inputs(
        self,
        expert_inputs: Dict[int, torch.Tensor],
        selected_experts: torch.Tensor
    ):
        """
        Analyze input patterns for each expert.
        
        Args:
            expert_inputs: Dictionary mapping expert indices to their inputs
            selected_experts: Selected expert indices
        """
        for expert_idx, inputs in expert_inputs.items():
            if expert_idx < self.num_experts:
                # Compute input statistics
                input_stats = {
                    'mean': inputs.mean().item(),
                    'std': inputs.std().item(),
                    'norm': torch.norm(inputs).item(),
                    'sparsity': (inputs == 0).float().mean().item()
                }
                self.expert_input_patterns[expert_idx].append(input_stats)
    
    def compute_specialization_scores(self) -> torch.Tensor:
        """
        Compute specialization scores for each expert.
        
        Returns:
            Specialization scores for each expert
        """
        for expert_idx in range(self.num_experts):
            if expert_idx in self.expert_input_patterns:
                patterns = self.expert_input_patterns[expert_idx]
                if len(patterns) > 1:
                    # Compute consistency of input patterns
                    mean_values = [p['mean'] for p in patterns]
                    std_values = [p['std'] for p in patterns]
                    
                    # High specialization = low variance in input patterns
                    mean_consistency = 1.0 / (1.0 + np.var(mean_values))
                    std_consistency = 1.0 / (1.0 + np.var(std_values))
                    
                    self.specialization_scores[expert_idx] = (mean_consistency + std_consistency) / 2
        
        return self.specialization_scores
    
    def get_specialization_report(self) -> Dict[str, Any]:
        """Get expert specialization report."""
        scores = self.compute_specialization_scores()
        
        report = {
            'specialization_scores': scores.tolist(),
            'most_specialized': int(torch.argmax(scores)),
            'least_specialized': int(torch.argmin(scores)),
            'average_specialization': scores.mean().item(),
            'specialization_variance': scores.var().item()
        }
        
        return report


class PerformanceProfiler:
    """Profiles performance impact of DenseMixer."""
    
    def __init__(self):
        """Initialize performance profiler."""
        self.timing_data = defaultdict(list)
        self.memory_usage = defaultdict(list)
        self.flops_estimates = defaultdict(list)
        self.current_timers = {}
    
    def start_timer(self, operation: str):
        """Start timing an operation."""
        self.current_timers[operation] = time.time()
    
    def end_timer(self, operation: str):
        """End timing an operation."""
        if operation in self.current_timers:
            elapsed = time.time() - self.current_timers[operation]
            self.timing_data[operation].append(elapsed)
            del self.current_timers[operation]
    
    def record_memory_usage(self, operation: str, memory_gb: float):
        """Record memory usage for an operation."""
        self.memory_usage[operation].append(memory_gb)
    
    def record_flops(self, operation: str, flops: float):
        """Record FLOPS for an operation."""
        self.flops_estimates[operation].append(flops)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        summary = {
            'timing': {},
            'memory': {},
            'flops': {}
        }
        
        # Timing statistics
        for operation, times in self.timing_data.items():
            summary['timing'][operation] = {
                'mean': float(np.mean(times)),
                'std': float(np.std(times)),
                'min': float(np.min(times)),
                'max': float(np.max(times)),
                'count': len(times)
            }
        
        # Memory statistics
        for operation, memory in self.memory_usage.items():
            summary['memory'][operation] = {
                'mean': float(np.mean(memory)),
                'std': float(np.std(memory)),
                'peak': float(np.max(memory)),
                'count': len(memory)
            }
        
        # FLOPS statistics
        for operation, flops in self.flops_estimates.items():
            summary['flops'][operation] = {
                'mean': float(np.mean(flops)),
                'total': float(np.sum(flops)),
                'count': len(flops)
            }
        
        return summary
    
    def compare_baseline(self, baseline_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Compare performance with baseline."""
        current_summary = self.get_performance_summary()
        comparison = {}
        
        # Compare timing
        for operation in current_summary['timing']:
            if operation in baseline_summary.get('timing', {}):
                current_time = current_summary['timing'][operation]['mean']
                baseline_time = baseline_summary['timing'][operation]['mean']
                overhead = (current_time - baseline_time) / baseline_time
                comparison[f'{operation}_timing_overhead'] = overhead
        
        # Compare memory
        for operation in current_summary['memory']:
            if operation in baseline_summary.get('memory', {}):
                current_memory = current_summary['memory'][operation]['mean']
                baseline_memory = baseline_summary['memory'][operation]['mean']
                overhead = (current_memory - baseline_memory) / baseline_memory
                comparison[f'{operation}_memory_overhead'] = overhead
        
        return comparison


def create_monitoring_suite(num_experts: int) -> Dict[str, Any]:
    """
    Create a complete monitoring suite for DenseMixer.
    
    Args:
        num_experts: Number of experts in the model
        
    Returns:
        Dictionary containing all monitoring components
    """
    suite = {
        'router_analytics': RouterAnalytics(num_experts),
        'specialization_analyzer': ExpertSpecializationAnalyzer(num_experts),
        'performance_profiler': PerformanceProfiler(),
    }
    
    return suite


def log_monitoring_summary(monitoring_suite: Dict[str, Any], logger: logging.Logger):
    """Log a summary of monitoring metrics."""
    router_analytics = monitoring_suite['router_analytics']
    specialization_analyzer = monitoring_suite['specialization_analyzer']
    performance_profiler = monitoring_suite['performance_profiler']
    
    # Router metrics
    metrics = router_analytics.get_current_metrics()
    logger.info(f"Router Status - Balance: {metrics.get('load_balance_coefficient', 0):.3f}, "
                f"Entropy: {metrics.get('routing_entropy', 0):.3f}, "
                f"Efficiency: {metrics.get('routing_efficiency', 0):.3f}")
    
    # Specialization
    spec_report = specialization_analyzer.get_specialization_report()
    logger.info(f"Expert Specialization - Average: {spec_report.get('average_specialization', 0):.3f}, "
                f"Most Specialized: Expert {spec_report.get('most_specialized', 0)}")
    
    # Performance
    perf_summary = performance_profiler.get_performance_summary()
    if 'forward_pass' in perf_summary['timing']:
        forward_time = perf_summary['timing']['forward_pass']['mean']
        logger.info(f"Performance - Forward Pass: {forward_time:.4f}s")
    
    # Issues
    issues = router_analytics.detect_routing_issues()
    if issues:
        logger.warning(f"Detected {len(issues)} routing issues")
        for issue in issues:
            logger.warning(f"  - {issue['type']}: {issue['description']}")