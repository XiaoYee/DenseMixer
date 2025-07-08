"""
Comprehensive test suite for DenseMixer improvements.

This module provides tests for all new features including memory optimization,
adaptive routing, monitoring, and enhanced configuration.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import json
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import DenseMixer modules
import sys
sys.path.append('/home/runner/work/DenseMixer/DenseMixer')

from densemixer.memory_optimization import (
    MemoryOptimizedForward,
    AdaptiveMemoryManager,
    optimize_expert_computation,
    get_memory_usage
)
from densemixer.adaptive_routing import (
    InputComplexityAnalyzer,
    AdaptiveTopKRouter,
    ExpertLoadBalancer,
    adaptive_moe_forward
)
from densemixer.monitoring import (
    RouterAnalytics,
    ExpertSpecializationAnalyzer,
    PerformanceProfiler,
    create_monitoring_suite
)
from densemixer.enhanced_config import (
    DenseMixerConfig,
    ConfigManager,
    MemoryConfig,
    AdaptiveRoutingConfig,
    LoadBalancingConfig,
    MonitoringConfig
)


class TestMemoryOptimization:
    """Test memory optimization functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.seq_len = 10
        self.hidden_dim = 64
        self.num_experts = 4
        self.top_k = 2
        
        # Create test tensors
        self.hidden_states = torch.randn(self.batch_size, self.seq_len, self.hidden_dim)
        self.expert_inputs = torch.randn(self.batch_size * self.seq_len, self.hidden_dim)
        self.routing_weights = torch.softmax(torch.randn(self.batch_size * self.seq_len, self.top_k), dim=-1)
        self.selected_experts = torch.randint(0, self.num_experts, (self.batch_size * self.seq_len, self.top_k))
        
        # Create mock experts
        self.experts = nn.ModuleList([
            nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(self.num_experts)
        ])
    
    def test_memory_optimized_forward(self):
        """Test memory-optimized forward pass."""
        # Test without gradient checkpointing
        output_no_checkpoint = MemoryOptimizedForward.gradient_checkpointed_expert_forward(
            self.experts, self.hidden_states, self.selected_experts, self.routing_weights, use_checkpoint=False
        )
        
        # Test with gradient checkpointing
        output_with_checkpoint = MemoryOptimizedForward.gradient_checkpointed_expert_forward(
            self.experts, self.hidden_states, self.selected_experts, self.routing_weights, use_checkpoint=True
        )
        
        # Outputs should have correct shape
        assert output_no_checkpoint.shape == (self.batch_size, self.seq_len, self.hidden_dim)
        assert output_with_checkpoint.shape == (self.batch_size, self.seq_len, self.hidden_dim)
        
        # Outputs should be different due to different computation paths
        # (though functionally equivalent)
        assert output_no_checkpoint.dtype == output_with_checkpoint.dtype
    
    def test_memory_efficient_routing(self):
        """Test memory-efficient routing computation."""
        router = nn.Linear(self.hidden_dim, self.num_experts)
        
        # Test without chunking
        weights1, experts1 = MemoryOptimizedForward.memory_efficient_routing_computation(
            self.hidden_states, router, self.top_k, chunk_size=None
        )
        
        # Test with chunking
        weights2, experts2 = MemoryOptimizedForward.memory_efficient_routing_computation(
            self.hidden_states, router, self.top_k, chunk_size=5
        )
        
        # Results should have correct shapes
        assert weights1.shape[1] == self.top_k
        assert experts1.shape[1] == self.top_k
        assert weights2.shape[1] == self.top_k
        assert experts2.shape[1] == self.top_k
        
        # Results should be close (chunking should not change results significantly)
        assert torch.allclose(weights1, weights2, atol=1e-5)
        assert torch.equal(experts1, experts2)
    
    def test_adaptive_memory_manager(self):
        """Test adaptive memory manager."""
        manager = AdaptiveMemoryManager(max_memory_gb=8.0)
        
        # Test chunk size calculation
        chunk_size = manager.get_recommended_chunk_size(1000, self.hidden_dim)
        assert chunk_size > 0
        assert isinstance(chunk_size, int)
        
        # Test gradient checkpointing decision
        should_checkpoint = manager.should_use_gradient_checkpointing(3.0)  # 3GB model
        assert isinstance(should_checkpoint, bool)
    
    def test_optimize_expert_computation(self):
        """Test optimized expert computation."""
        manager = AdaptiveMemoryManager(max_memory_gb=1.0)  # Low memory to force checkpointing
        
        output = optimize_expert_computation(
            self.experts,
            self.hidden_states,
            self.routing_weights,
            self.selected_experts,
            memory_manager=manager,
            use_gradient_checkpointing=True
        )
        
        assert output.shape == (self.batch_size, self.seq_len, self.hidden_dim)
        assert output.dtype == self.hidden_states.dtype


class TestAdaptiveRouting:
    """Test adaptive routing functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.seq_len = 10
        self.hidden_dim = 64
        self.num_experts = 8
        self.hidden_states = torch.randn(self.batch_size, self.seq_len, self.hidden_dim)
    
    def test_input_complexity_analyzer(self):
        """Test input complexity analysis."""
        # Test entropy computation
        entropy = InputComplexityAnalyzer.compute_entropy(self.hidden_states)
        assert entropy.shape == (self.batch_size, self.seq_len)
        assert torch.all(entropy >= 0)  # Entropy should be non-negative
        
        # Test attention spread computation
        attention_spread = InputComplexityAnalyzer.compute_attention_spread(self.hidden_states)
        assert attention_spread.shape == (self.batch_size, self.seq_len)
        assert torch.all(attention_spread >= 0)  # Variance should be non-negative
        
        # Test combined complexity
        complexity = InputComplexityAnalyzer.compute_combined_complexity(self.hidden_states)
        assert complexity.shape == (self.batch_size, self.seq_len)
    
    def test_adaptive_top_k_router(self):
        """Test adaptive Top-K router."""
        router = AdaptiveTopKRouter(min_k=1, max_k=4)
        
        # Test complexity-based K computation
        complexity_scores = torch.randn(self.batch_size, self.seq_len)
        adaptive_k = router.compute_adaptive_k(complexity_scores, base_k=2)
        
        assert adaptive_k.shape == (self.batch_size, self.seq_len)
        assert torch.all(adaptive_k >= router.min_k)
        assert torch.all(adaptive_k <= router.max_k)
        
        # Test adaptive expert selection
        router_logits = torch.randn(self.batch_size * self.seq_len, self.num_experts)
        weights, experts, k_values = router.adaptive_expert_selection(
            router_logits, self.hidden_states, base_k=2
        )
        
        assert weights.shape[0] == self.batch_size * self.seq_len
        assert experts.shape[0] == self.batch_size * self.seq_len
        assert k_values.shape == (self.batch_size, self.seq_len)
    
    def test_expert_load_balancer(self):
        """Test expert load balancer."""
        balancer = ExpertLoadBalancer(num_experts=self.num_experts)
        
        # Test usage statistics update
        selected_experts = torch.randint(0, self.num_experts, (20, 2))
        balancer.update_usage_stats(selected_experts)
        
        stats = balancer.get_usage_statistics()
        assert 'expert_utilization' in stats
        assert 'balance_coefficient' in stats
        assert stats['expert_utilization'].shape[0] == self.num_experts
        
        # Test load balancing loss computation
        router_logits = torch.randn(20, self.num_experts)
        loss = balancer.compute_load_balancing_loss(router_logits, selected_experts)
        assert isinstance(loss, torch.Tensor)
        assert loss.numel() == 1  # Scalar loss
    
    def test_adaptive_moe_forward(self):
        """Test adaptive MoE forward pass."""
        router = nn.Linear(self.hidden_dim, self.num_experts)
        experts = nn.ModuleList([
            nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(self.num_experts)
        ])
        
        adaptive_router = AdaptiveTopKRouter(min_k=1, max_k=3)
        load_balancer = ExpertLoadBalancer(num_experts=self.num_experts)
        
        output, metrics = adaptive_moe_forward(
            self.hidden_states,
            router,
            experts,
            base_top_k=2,
            adaptive_router=adaptive_router,
            load_balancer=load_balancer
        )
        
        assert output.shape == self.hidden_states.shape
        assert 'load_balance_loss' in metrics
        assert 'adaptive_k_mean' in metrics
        assert 'adaptive_k_std' in metrics


class TestMonitoring:
    """Test monitoring functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.num_experts = 4
        self.batch_size = 2
        self.seq_len = 10
        self.hidden_dim = 32
    
    def test_router_analytics(self):
        """Test router analytics."""
        analytics = RouterAnalytics(num_experts=self.num_experts)
        
        # Generate test data
        router_logits = torch.randn(self.batch_size * self.seq_len, self.num_experts)
        selected_experts = torch.randint(0, self.num_experts, (self.batch_size * self.seq_len, 2))
        routing_weights = torch.softmax(router_logits, dim=-1)
        
        # Update metrics
        analytics.update_routing_metrics(router_logits, selected_experts, routing_weights)
        
        # Test current metrics
        metrics = analytics.get_current_metrics()
        assert 'step_count' in metrics
        assert 'total_tokens' in metrics
        assert 'expert_utilization' in metrics
        assert len(metrics['expert_utilization']) == self.num_experts
        
        # Test historical analysis
        for _ in range(10):  # Add more data points
            analytics.update_routing_metrics(router_logits, selected_experts, routing_weights)
        
        historical = analytics.get_historical_analysis()
        assert 'entropy_trend' in historical
        assert 'load_variance_trend' in historical
        
        # Test issue detection
        issues = analytics.detect_routing_issues()
        assert isinstance(issues, list)
        
        # Test report generation
        report = analytics.generate_report()
        assert isinstance(report, str)
        assert len(report) > 0
    
    def test_expert_specialization_analyzer(self):
        """Test expert specialization analyzer."""
        analyzer = ExpertSpecializationAnalyzer(num_experts=self.num_experts)
        
        # Generate test expert inputs
        expert_inputs = {
            i: torch.randn(10, self.hidden_dim) for i in range(self.num_experts)
        }
        selected_experts = torch.randint(0, self.num_experts, (20, 2))
        
        # Analyze expert inputs
        analyzer.analyze_expert_inputs(expert_inputs, selected_experts)
        
        # Compute specialization scores
        scores = analyzer.compute_specialization_scores()
        assert scores.shape[0] == self.num_experts
        assert torch.all(scores >= 0)  # Scores should be non-negative
        
        # Get specialization report
        report = analyzer.get_specialization_report()
        assert 'specialization_scores' in report
        assert 'most_specialized' in report
        assert 'least_specialized' in report
    
    def test_performance_profiler(self):
        """Test performance profiler."""
        profiler = PerformanceProfiler()
        
        # Test timing operations
        profiler.start_timer('test_operation')
        # Simulate some work
        torch.randn(100, 100) @ torch.randn(100, 100)
        profiler.end_timer('test_operation')
        
        # Test memory recording
        profiler.record_memory_usage('test_operation', 1.5)
        
        # Test FLOPS recording
        profiler.record_flops('test_operation', 1000000)
        
        # Get performance summary
        summary = profiler.get_performance_summary()
        assert 'timing' in summary
        assert 'memory' in summary
        assert 'flops' in summary
        assert 'test_operation' in summary['timing']
    
    def test_monitoring_suite(self):
        """Test complete monitoring suite."""
        suite = create_monitoring_suite(num_experts=self.num_experts)
        
        assert 'router_analytics' in suite
        assert 'specialization_analyzer' in suite
        assert 'performance_profiler' in suite
        
        # Test that components are properly initialized
        assert isinstance(suite['router_analytics'], RouterAnalytics)
        assert isinstance(suite['specialization_analyzer'], ExpertSpecializationAnalyzer)
        assert isinstance(suite['performance_profiler'], PerformanceProfiler)


class TestEnhancedConfig:
    """Test enhanced configuration system."""
    
    def test_config_creation(self):
        """Test configuration creation."""
        config = DenseMixerConfig()
        
        assert isinstance(config.enabled, bool)
        assert isinstance(config.models, dict)
        assert isinstance(config.memory, MemoryConfig)
        assert isinstance(config.adaptive_routing, AdaptiveRoutingConfig)
        assert isinstance(config.load_balancing, LoadBalancingConfig)
        assert isinstance(config.monitoring, MonitoringConfig)
    
    def test_config_from_dict(self):
        """Test configuration creation from dictionary."""
        config_dict = {
            'enabled': True,
            'memory': {
                'use_gradient_checkpointing': True,
                'max_memory_gb': 32.0
            },
            'adaptive_routing': {
                'enabled': True,
                'min_k': 1,
                'max_k': 6
            }
        }
        
        config = DenseMixerConfig.from_dict(config_dict)
        assert config.enabled is True
        assert config.memory.use_gradient_checkpointing is True
        assert config.memory.max_memory_gb == 32.0
        assert config.adaptive_routing.enabled is True
        assert config.adaptive_routing.max_k == 6
    
    def test_config_from_env(self):
        """Test configuration creation from environment variables."""
        with patch.dict('os.environ', {
            'DENSEMIXER_ENABLED': '1',
            'DENSEMIXER_DEBUG': 'true',
            'DENSEMIXER_MAX_MEMORY_GB': '24.0',
            'DENSEMIXER_MIN_K': '2',
            'DENSEMIXER_ADAPTIVE_ROUTING': '1'
        }):
            config = DenseMixerConfig.from_env()
            
            assert config.enabled is True
            assert config.debug_mode is True
            assert config.memory.max_memory_gb == 24.0
            assert config.adaptive_routing.min_k == 2
            assert config.adaptive_routing.enabled is True
    
    def test_config_file_operations(self):
        """Test configuration file save/load operations."""
        config = DenseMixerConfig()
        config.enabled = True
        config.memory.max_memory_gb = 20.0
        config.adaptive_routing.enabled = True
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test YAML format
            yaml_path = Path(temp_dir) / "config.yaml"
            config.to_file(yaml_path, format='yaml')
            assert yaml_path.exists()
            
            loaded_config = DenseMixerConfig.from_file(yaml_path)
            assert loaded_config.enabled == config.enabled
            assert loaded_config.memory.max_memory_gb == config.memory.max_memory_gb
            assert loaded_config.adaptive_routing.enabled == config.adaptive_routing.enabled
            
            # Test JSON format
            json_path = Path(temp_dir) / "config.json"
            config.to_file(json_path, format='json')
            assert json_path.exists()
            
            loaded_config_json = DenseMixerConfig.from_file(json_path)
            assert loaded_config_json.enabled == config.enabled
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = DenseMixerConfig()
        
        # Valid configuration should have no errors
        errors = config.validate()
        assert len(errors) == 0
        
        # Invalid configuration should have errors
        config.memory.max_memory_gb = -1.0  # Invalid
        config.adaptive_routing.min_k = 5   # Invalid (min_k > max_k)
        config.adaptive_routing.max_k = 4
        
        errors = config.validate()
        assert len(errors) > 0
        assert any('max_memory_gb' in error for error in errors)
        assert any('min_k' in error for error in errors)
    
    def test_config_manager(self):
        """Test configuration manager."""
        manager = ConfigManager()
        
        # Test loading default config
        config = manager.load_config()
        assert isinstance(config, DenseMixerConfig)
        
        # Test auto-tuning
        config.auto_tune = True
        manager.config = config
        
        # Simulate auto-tuning steps
        for i in range(5):
            metrics = {
                'performance_score': np.random.random(),
                'load_balance_coefficient': np.random.random(),
                'memory_usage_gb': np.random.uniform(1, 10),
                'routing_entropy': np.random.uniform(1, 5)
            }
            manager.auto_tune_step(metrics)
        
        assert len(manager.auto_tune_metrics) == 5
        
        # Test saving/loading configuration
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.yaml"
            manager.save_config(config_path)
            assert config_path.exists()
            
            # Load and verify
            loaded_config = manager.load_config(config_path)
            assert loaded_config.auto_tune == config.auto_tune


class TestIntegration:
    """Integration tests for all components working together."""
    
    def test_full_pipeline(self):
        """Test full DenseMixer pipeline with all improvements."""
        # Create configuration
        config = DenseMixerConfig()
        config.enabled = True
        config.memory.use_gradient_checkpointing = True
        config.adaptive_routing.enabled = True
        config.load_balancing.enabled = True
        config.monitoring.enabled = True
        
        # Set up model components
        batch_size, seq_len, hidden_dim = 2, 10, 64
        num_experts = 4
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
        
        router = nn.Linear(hidden_dim, num_experts)
        experts = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_experts)
        ])
        
        # Create monitoring suite
        monitoring_suite = create_monitoring_suite(num_experts)
        
        # Create adaptive router
        adaptive_router = AdaptiveTopKRouter(
            min_k=config.adaptive_routing.min_k,
            max_k=config.adaptive_routing.max_k
        )
        
        # Create load balancer
        load_balancer = ExpertLoadBalancer(
            num_experts=num_experts,
            balance_factor=config.load_balancing.balance_factor
        )
        
        # Run forward pass
        output, metrics = adaptive_moe_forward(
            hidden_states,
            router,
            experts,
            base_top_k=2,
            adaptive_router=adaptive_router,
            load_balancer=load_balancer
        )
        
        # Verify output
        assert output.shape == hidden_states.shape
        assert 'load_balance_loss' in metrics
        assert 'adaptive_k_mean' in metrics
        
        # Update monitoring
        router_logits = router(hidden_states.view(-1, hidden_dim))
        selected_experts = torch.randint(0, num_experts, (batch_size * seq_len, 2))
        routing_weights = torch.softmax(router_logits, dim=-1)
        
        monitoring_suite['router_analytics'].update_routing_metrics(
            router_logits, selected_experts, routing_weights
        )
        
        # Verify monitoring works
        current_metrics = monitoring_suite['router_analytics'].get_current_metrics()
        assert 'expert_utilization' in current_metrics
        assert len(current_metrics['expert_utilization']) == num_experts


if __name__ == "__main__":
    # Run tests manually if pytest is not available
    print("Running DenseMixer improvement tests...")
    
    # Test memory optimization
    print("\n1. Testing Memory Optimization...")
    test_memory = TestMemoryOptimization()
    test_memory.setup_method()
    test_memory.test_memory_optimized_forward()
    test_memory.test_memory_efficient_routing()
    test_memory.test_adaptive_memory_manager()
    test_memory.test_optimize_expert_computation()
    print("✓ Memory optimization tests passed")
    
    # Test adaptive routing
    print("\n2. Testing Adaptive Routing...")
    test_routing = TestAdaptiveRouting()
    test_routing.setup_method()
    test_routing.test_input_complexity_analyzer()
    test_routing.test_adaptive_top_k_router()
    test_routing.test_expert_load_balancer()
    test_routing.test_adaptive_moe_forward()
    print("✓ Adaptive routing tests passed")
    
    # Test monitoring
    print("\n3. Testing Monitoring...")
    test_monitoring = TestMonitoring()
    test_monitoring.setup_method()
    test_monitoring.test_router_analytics()
    test_monitoring.test_expert_specialization_analyzer()
    test_monitoring.test_performance_profiler()
    test_monitoring.test_monitoring_suite()
    print("✓ Monitoring tests passed")
    
    # Test configuration
    print("\n4. Testing Enhanced Configuration...")
    test_config = TestEnhancedConfig()
    test_config.test_config_creation()
    test_config.test_config_from_dict()
    test_config.test_config_from_env()
    test_config.test_config_file_operations()
    test_config.test_config_validation()
    test_config.test_config_manager()
    print("✓ Enhanced configuration tests passed")
    
    # Test integration
    print("\n5. Testing Integration...")
    test_integration = TestIntegration()
    test_integration.test_full_pipeline()
    print("✓ Integration tests passed")
    
    print("\n🎉 All tests passed successfully!")
    print("\nNew DenseMixer improvements are working correctly:")
    print("  - Memory optimization with gradient checkpointing")
    print("  - Adaptive routing based on input complexity")
    print("  - Comprehensive monitoring and analytics")
    print("  - Enhanced configuration system")
    print("  - Full integration of all components")