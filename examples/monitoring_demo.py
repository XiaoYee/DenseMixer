#!/usr/bin/env python3
"""
Example script for monitoring DenseMixer router behavior during training.

This script demonstrates how to use the comprehensive monitoring system
to track router behavior, expert utilization, and performance metrics.
"""

import torch
import torch.nn as nn
import logging
from pathlib import Path
import json
import sys

# Add DenseMixer to path
sys.path.append(str(Path(__file__).parent.parent))

from densemixer.monitoring import (
    RouterAnalytics, 
    ExpertSpecializationAnalyzer, 
    PerformanceProfiler,
    create_monitoring_suite,
    log_monitoring_summary
)
from densemixer.adaptive_routing import AdaptiveTopKRouter, ExpertLoadBalancer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MockMoETrainer:
    """Mock MoE trainer to demonstrate monitoring functionality."""
    
    def __init__(self, num_experts=8, hidden_dim=256):
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.top_k = 2
        
        # Create mock model components
        self.router = nn.Linear(hidden_dim, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim)
            ) for _ in range(num_experts)
        ])
        
        # Create monitoring suite
        self.monitoring_suite = create_monitoring_suite(num_experts)
        
        # Create adaptive components
        self.adaptive_router = AdaptiveTopKRouter(min_k=1, max_k=4)
        self.load_balancer = ExpertLoadBalancer(num_experts, balance_factor=0.01)
        
        logger.info(f"Created mock MoE trainer with {num_experts} experts")
    
    def simulate_training_step(self, step: int):
        """Simulate one training step."""
        # Generate random input
        batch_size, seq_len = 4, 16
        hidden_states = torch.randn(batch_size, seq_len, self.hidden_dim)
        flat_hidden = hidden_states.view(-1, self.hidden_dim)
        
        # Router computation
        router_logits = self.router(flat_hidden)
        routing_weights = torch.softmax(router_logits, dim=-1)
        routing_weights_topk, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        
        # Simulate gradients for analysis
        if step > 5:  # Start computing gradients after a few steps
            router_logits.retain_grad()
            loss = torch.sum(routing_weights)
            loss.backward(retain_graph=True)
            router_gradients = router_logits.grad
        else:
            router_gradients = None
        
        # Update monitoring
        self.monitoring_suite['router_analytics'].update_routing_metrics(
            router_logits.detach(),
            selected_experts,
            routing_weights.detach(),
            router_gradients.detach() if router_gradients is not None else None
        )
        
        # Update load balancer
        self.load_balancer.update_usage_stats(selected_experts)
        
        # Performance profiling
        profiler = self.monitoring_suite['performance_profiler']
        profiler.start_timer('forward_pass')
        
        # Simulate forward pass timing
        torch.randn(1000, 1000) @ torch.randn(1000, 1000)
        
        profiler.end_timer('forward_pass')
        profiler.record_memory_usage('forward_pass', torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 2.5)
        profiler.record_flops('forward_pass', batch_size * seq_len * self.hidden_dim * self.num_experts * 2)
        
        # Expert specialization analysis
        if step % 10 == 0:
            expert_inputs = {}
            for expert_idx in range(self.num_experts):
                expert_mask = (selected_experts == expert_idx).any(dim=-1)
                if expert_mask.any():
                    expert_inputs[expert_idx] = flat_hidden[expert_mask]
            
            self.monitoring_suite['specialization_analyzer'].analyze_expert_inputs(
                expert_inputs, selected_experts
            )
        
        return {
            'router_logits': router_logits.detach(),
            'selected_experts': selected_experts,
            'routing_weights': routing_weights.detach(),
            'load_balance_loss': self.load_balancer.compute_load_balancing_loss(router_logits.detach(), selected_experts)
        }
    
    def run_training_simulation(self, num_steps=100):
        """Run a full training simulation with monitoring."""
        logger.info(f"Starting training simulation for {num_steps} steps...")
        
        all_metrics = []
        
        for step in range(num_steps):
            # Simulate training step
            step_metrics = self.simulate_training_step(step)
            
            # Log monitoring summary periodically
            if step % 20 == 0 and step > 0:
                log_monitoring_summary(self.monitoring_suite, logger)
            
            # Collect metrics for analysis
            if step % 10 == 0:
                current_metrics = self.monitoring_suite['router_analytics'].get_current_metrics()
                all_metrics.append({
                    'step': step,
                    **current_metrics
                })
        
        return all_metrics
    
    def generate_final_report(self):
        """Generate comprehensive final report."""
        logger.info("\n" + "="*50)
        logger.info("DENSEMIXER MONITORING FINAL REPORT")
        logger.info("="*50)
        
        # Router analytics report
        analytics = self.monitoring_suite['router_analytics']
        router_report = analytics.generate_report()
        print(router_report)
        
        # Expert specialization report
        specialization_analyzer = self.monitoring_suite['specialization_analyzer']
        spec_report = specialization_analyzer.get_specialization_report()
        
        logger.info("\n--- Expert Specialization Analysis ---")
        logger.info(f"Average specialization: {spec_report.get('average_specialization', 0):.3f}")
        logger.info(f"Most specialized expert: {spec_report.get('most_specialized', 'N/A')}")
        logger.info(f"Least specialized expert: {spec_report.get('least_specialized', 'N/A')}")
        logger.info(f"Specialization variance: {spec_report.get('specialization_variance', 0):.3f}")
        
        # Performance profiling report
        profiler = self.monitoring_suite['performance_profiler']
        perf_summary = profiler.get_performance_summary()
        
        logger.info("\n--- Performance Analysis ---")
        if 'forward_pass' in perf_summary.get('timing', {}):
            timing = perf_summary['timing']['forward_pass']
            logger.info(f"Forward pass timing - Mean: {timing['mean']:.4f}s, Std: {timing['std']:.4f}s")
        
        if 'forward_pass' in perf_summary.get('memory', {}):
            memory = perf_summary['memory']['forward_pass']
            logger.info(f"Memory usage - Mean: {memory['mean']:.2f}GB, Peak: {memory['peak']:.2f}GB")
        
        if 'forward_pass' in perf_summary.get('flops', {}):
            flops = perf_summary['flops']['forward_pass']
            logger.info(f"FLOPS - Total: {flops['total']:.2e}, Mean per step: {flops['mean']:.2e}")
        
        # Issue detection
        issues = analytics.detect_routing_issues()
        if issues:
            logger.warning(f"\n--- Detected Issues ({len(issues)}) ---")
            for issue in issues:
                logger.warning(f"  {issue['type']} ({issue['severity']}): {issue['description']}")
        else:
            logger.info("\n--- No routing issues detected ---")
        
        # Load balancing statistics
        load_stats = self.load_balancer.get_usage_statistics()
        logger.info("\n--- Load Balancing Statistics ---")
        logger.info(f"Balance coefficient: {load_stats.get('balance_coefficient', 0):.3f}")
        logger.info(f"Total tokens processed: {load_stats.get('total_tokens_processed', 0)}")
        
        utilization = load_stats.get('expert_utilization', [])
        if len(utilization) > 0:
            logger.info("Expert utilization:")
            for i, util in enumerate(utilization):
                logger.info(f"  Expert {i}: {util:.1%}")
    
    def save_metrics_to_file(self, all_metrics, filename="monitoring_results.json"):
        """Save collected metrics to file."""
        output_path = Path(filename)
        
        # Prepare data for JSON serialization
        serializable_metrics = []
        for metric in all_metrics:
            serializable_metric = {}
            for key, value in metric.items():
                if isinstance(value, (list, tuple)):
                    serializable_metric[key] = [float(x) if hasattr(x, 'item') else x for x in value]
                elif hasattr(value, 'item'):
                    serializable_metric[key] = value.item()
                else:
                    serializable_metric[key] = value
            serializable_metrics.append(serializable_metric)
        
        # Add summary statistics
        final_report = {
            'metrics_history': serializable_metrics,
            'expert_specialization': self.monitoring_suite['specialization_analyzer'].get_specialization_report(),
            'performance_summary': self.monitoring_suite['performance_profiler'].get_performance_summary(),
            'router_issues': self.monitoring_suite['router_analytics'].detect_routing_issues(),
            'load_balancing': self.load_balancer.get_usage_statistics()
        }
        
        # Convert any remaining tensors to lists
        def convert_tensors(obj):
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_tensors(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_tensors(item) for item in obj]
            else:
                return obj
        
        final_report = convert_tensors(final_report)
        
        with open(output_path, 'w') as f:
            json.dump(final_report, f, indent=2)
        
        logger.info(f"Saved monitoring results to {output_path}")


def main():
    """Main function to run monitoring demonstration."""
    logger.info("Starting DenseMixer monitoring demonstration...")
    
    # Create trainer with monitoring
    trainer = MockMoETrainer(num_experts=8, hidden_dim=256)
    
    # Run training simulation
    all_metrics = trainer.run_training_simulation(num_steps=100)
    
    # Generate final report
    trainer.generate_final_report()
    
    # Save results
    trainer.save_metrics_to_file(all_metrics)
    
    logger.info("\n🎉 Monitoring demonstration completed successfully!")
    logger.info("\nKey monitoring features demonstrated:")
    logger.info("  ✓ Real-time router analytics")
    logger.info("  ✓ Expert utilization tracking")
    logger.info("  ✓ Performance profiling")
    logger.info("  ✓ Specialization analysis")
    logger.info("  ✓ Issue detection")
    logger.info("  ✓ Load balancing statistics")
    logger.info("  ✓ Comprehensive reporting")


if __name__ == "__main__":
    main()