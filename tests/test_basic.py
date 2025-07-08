"""
Simplified test for DenseMixer improvements to verify core functionality.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import tempfile
from pathlib import Path

# Add DenseMixer to path
sys.path.append('/home/runner/work/DenseMixer/DenseMixer')

def test_memory_optimization():
    """Test basic memory optimization functionality."""
    print("Testing memory optimization...")
    
    from densemixer.memory_optimization import AdaptiveMemoryManager, get_memory_usage
    
    # Test memory manager
    manager = AdaptiveMemoryManager(max_memory_gb=8.0)
    
    # Test chunk size calculation
    chunk_size = manager.get_recommended_chunk_size(1000, 64)
    assert chunk_size > 0
    assert isinstance(chunk_size, int)
    
    # Test gradient checkpointing decision
    should_checkpoint = manager.should_use_gradient_checkpointing(3.0)
    assert isinstance(should_checkpoint, bool)
    
    # Test memory usage function
    memory_stats = get_memory_usage()
    assert isinstance(memory_stats, dict)
    assert 'allocated_gb' in memory_stats
    
    print("✓ Memory optimization basic tests passed")


def test_adaptive_routing():
    """Test basic adaptive routing functionality."""
    print("Testing adaptive routing...")
    
    from densemixer.adaptive_routing import InputComplexityAnalyzer, AdaptiveTopKRouter, ExpertLoadBalancer
    
    batch_size, seq_len, hidden_dim = 2, 10, 64
    num_experts = 4
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
    
    # Test complexity analyzer
    entropy = InputComplexityAnalyzer.compute_entropy(hidden_states)
    assert entropy.shape == (batch_size, seq_len)
    assert torch.all(entropy >= 0)
    
    attention_spread = InputComplexityAnalyzer.compute_attention_spread(hidden_states)
    assert attention_spread.shape == (batch_size, seq_len)
    assert torch.all(attention_spread >= 0)
    
    # Test adaptive router
    router = AdaptiveTopKRouter(min_k=1, max_k=4)
    complexity_scores = torch.randn(batch_size, seq_len)
    adaptive_k = router.compute_adaptive_k(complexity_scores, base_k=2)
    
    assert adaptive_k.shape == (batch_size, seq_len)
    assert torch.all(adaptive_k >= router.min_k)
    assert torch.all(adaptive_k <= router.max_k)
    
    # Test load balancer
    balancer = ExpertLoadBalancer(num_experts=num_experts)
    selected_experts = torch.randint(0, num_experts, (20, 2))
    balancer.update_usage_stats(selected_experts)
    
    stats = balancer.get_usage_statistics()
    assert 'expert_utilization' in stats
    assert 'balance_coefficient' in stats
    
    print("✓ Adaptive routing basic tests passed")


def test_monitoring():
    """Test basic monitoring functionality."""
    print("Testing monitoring...")
    
    from densemixer.monitoring import RouterAnalytics, PerformanceProfiler, create_monitoring_suite
    
    num_experts = 4
    batch_size, seq_len = 2, 10
    
    # Test router analytics
    analytics = RouterAnalytics(num_experts=num_experts)
    
    router_logits = torch.randn(batch_size * seq_len, num_experts)
    selected_experts = torch.randint(0, num_experts, (batch_size * seq_len, 2))
    routing_weights = torch.softmax(router_logits, dim=-1)
    
    analytics.update_routing_metrics(router_logits, selected_experts, routing_weights)
    
    metrics = analytics.get_current_metrics()
    assert 'step_count' in metrics
    assert 'total_tokens' in metrics
    assert 'expert_utilization' in metrics
    
    # Test performance profiler
    profiler = PerformanceProfiler()
    
    profiler.start_timer('test_operation')
    torch.randn(100, 100) @ torch.randn(100, 100)  # Simulate work
    profiler.end_timer('test_operation')
    
    profiler.record_memory_usage('test_operation', 1.5)
    profiler.record_flops('test_operation', 1000000)
    
    summary = profiler.get_performance_summary()
    assert 'timing' in summary
    assert 'memory' in summary
    assert 'flops' in summary
    
    # Test monitoring suite
    suite = create_monitoring_suite(num_experts=num_experts)
    assert 'router_analytics' in suite
    assert 'specialization_analyzer' in suite
    assert 'performance_profiler' in suite
    
    print("✓ Monitoring basic tests passed")


def test_enhanced_config():
    """Test basic configuration functionality."""
    print("Testing enhanced configuration...")
    
    from densemixer.enhanced_config import DenseMixerConfig, ConfigManager
    
    # Test config creation
    config = DenseMixerConfig()
    assert isinstance(config.enabled, bool)
    assert isinstance(config.models, dict)
    
    # Test config from dict
    config_dict = {
        'enabled': True,
        'memory': {
            'use_gradient_checkpointing': True,
            'max_memory_gb': 32.0
        }
    }
    
    config = DenseMixerConfig.from_dict(config_dict)
    assert config.enabled is True
    assert config.memory.use_gradient_checkpointing is True
    assert config.memory.max_memory_gb == 32.0
    
    # Test validation
    errors = config.validate()
    assert len(errors) == 0  # Should be valid
    
    # Test invalid config
    config.memory.max_memory_gb = -1.0
    errors = config.validate()
    assert len(errors) > 0  # Should have errors
    
    # Test config manager
    manager = ConfigManager()
    config = manager.load_config()
    assert isinstance(config, DenseMixerConfig)
    
    # Test file operations
    config = DenseMixerConfig()
    config.enabled = True
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "test_config.yaml"
        config.to_file(config_path)
        assert config_path.exists()
        
        loaded_config = DenseMixerConfig.from_file(config_path)
        assert loaded_config.enabled == config.enabled
    
    print("✓ Enhanced configuration basic tests passed")


def test_integration():
    """Test basic integration of components."""
    print("Testing integration...")
    
    from densemixer.enhanced_config import DenseMixerConfig
    from densemixer.adaptive_routing import AdaptiveTopKRouter, ExpertLoadBalancer
    from densemixer.monitoring import create_monitoring_suite
    
    # Create configuration
    config = DenseMixerConfig()
    config.enabled = True
    config.adaptive_routing.enabled = True
    config.load_balancing.enabled = True
    config.monitoring.enabled = True
    
    # Set up simple model components
    batch_size, seq_len, hidden_dim = 2, 10, 64
    num_experts = 4
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
    
    router = nn.Linear(hidden_dim, num_experts)
    experts = nn.ModuleList([
        nn.Linear(hidden_dim, hidden_dim) for _ in range(num_experts)
    ])
    
    # Create components
    monitoring_suite = create_monitoring_suite(num_experts)
    adaptive_router = AdaptiveTopKRouter(
        min_k=config.adaptive_routing.min_k,
        max_k=config.adaptive_routing.max_k
    )
    load_balancer = ExpertLoadBalancer(
        num_experts=num_experts,
        balance_factor=config.load_balancing.balance_factor
    )
    
    # Simple forward pass simulation
    flat_hidden = hidden_states.view(-1, hidden_dim)
    router_logits = router(flat_hidden)
    routing_weights = torch.softmax(router_logits, dim=-1)
    routing_weights_topk, selected_experts = torch.topk(routing_weights, 2, dim=-1)
    
    # Test monitoring update
    monitoring_suite['router_analytics'].update_routing_metrics(
        router_logits, selected_experts, routing_weights
    )
    
    # Verify monitoring works
    current_metrics = monitoring_suite['router_analytics'].get_current_metrics()
    assert 'expert_utilization' in current_metrics
    assert len(current_metrics['expert_utilization']) == num_experts
    
    print("✓ Integration basic tests passed")


if __name__ == "__main__":
    print("Running simplified DenseMixer improvement tests...\n")
    
    try:
        test_memory_optimization()
        test_adaptive_routing()
        test_monitoring()
        test_enhanced_config()
        test_integration()
        
        print("\n🎉 All basic tests passed successfully!")
        print("\nDenseMixer improvements are working correctly:")
        print("  ✓ Memory optimization infrastructure")
        print("  ✓ Adaptive routing with complexity analysis")
        print("  ✓ Comprehensive monitoring and analytics")
        print("  ✓ Enhanced configuration system")
        print("  ✓ Component integration")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)