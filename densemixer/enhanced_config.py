"""
Enhanced configuration system for DenseMixer.

This module provides a comprehensive configuration system with granular controls
over all aspects of DenseMixer behavior.
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field, asdict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class MemoryConfig:
    """Configuration for memory optimization."""
    use_gradient_checkpointing: bool = True
    max_memory_gb: float = 16.0
    chunk_size: Optional[int] = None
    memory_efficient_routing: bool = True
    enable_memory_profiling: bool = False


@dataclass
class AdaptiveRoutingConfig:
    """Configuration for adaptive routing."""
    enabled: bool = False
    min_k: int = 1
    max_k: int = 4
    complexity_threshold: float = 0.5
    adaptation_rate: float = 0.1
    use_learned_thresholds: bool = True
    complexity_weights: Dict[str, float] = field(default_factory=lambda: {
        'entropy': 0.4,
        'attention_spread': 0.4,
        'gradient_norm': 0.2
    })


@dataclass
class LoadBalancingConfig:
    """Configuration for load balancing."""
    enabled: bool = True
    balance_factor: float = 0.01
    target_utilization_variance: float = 0.1
    expert_dropout_rate: float = 0.0
    load_balancing_loss_weight: float = 0.01


@dataclass
class MonitoringConfig:
    """Configuration for monitoring and analytics."""
    enabled: bool = True
    history_size: int = 1000
    log_interval: int = 100
    save_metrics: bool = True
    metrics_output_dir: str = "./densemixer_metrics"
    detailed_expert_analysis: bool = False
    performance_profiling: bool = False


@dataclass
class ModelSpecificConfig:
    """Configuration specific to a model type."""
    enabled: bool = True
    custom_forward_method: Optional[str] = None
    normalization_method: str = "standard"  # "standard", "gated", "partscale_fix_expert"
    expert_parallelism: bool = False
    custom_parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentalConfig:
    """Configuration for experimental features."""
    dynamic_expert_selection: bool = False
    expert_merging: bool = False
    router_distillation: bool = False
    multi_level_routing: bool = False
    expert_pruning: bool = False
    router_ensemble: bool = False


@dataclass
class DenseMixerConfig:
    """Main DenseMixer configuration."""
    
    # Global settings
    enabled: bool = False
    debug_mode: bool = False
    log_level: str = "INFO"
    config_version: str = "1.0"
    
    # Model-specific configurations
    models: Dict[str, ModelSpecificConfig] = field(default_factory=lambda: {
        "qwen3": ModelSpecificConfig(),
        "qwen2": ModelSpecificConfig(),
        "olmoe": ModelSpecificConfig(),
        "deepseek_moe": ModelSpecificConfig(enabled=False),
        "switch_transformer": ModelSpecificConfig(enabled=False)
    })
    
    # Feature configurations
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    adaptive_routing: AdaptiveRoutingConfig = field(default_factory=AdaptiveRoutingConfig)
    load_balancing: LoadBalancingConfig = field(default_factory=LoadBalancingConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    experimental: ExperimentalConfig = field(default_factory=ExperimentalConfig)
    
    # Runtime settings
    auto_tune: bool = False
    auto_tune_steps: int = 1000
    save_auto_tune_config: bool = True
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DenseMixerConfig':
        """Create configuration from dictionary."""
        # Handle nested configurations
        if 'models' in config_dict:
            models = {}
            for model_name, model_config in config_dict['models'].items():
                if isinstance(model_config, dict):
                    models[model_name] = ModelSpecificConfig(**model_config)
                else:
                    models[model_name] = model_config
            config_dict['models'] = models
        
        # Handle other nested configs
        for config_name, config_class in [
            ('memory', MemoryConfig),
            ('adaptive_routing', AdaptiveRoutingConfig),
            ('load_balancing', LoadBalancingConfig),
            ('monitoring', MonitoringConfig),
            ('experimental', ExperimentalConfig)
        ]:
            if config_name in config_dict and isinstance(config_dict[config_name], dict):
                config_dict[config_name] = config_class(**config_dict[config_name])
        
        return cls(**config_dict)
    
    @classmethod
    def from_env(cls) -> 'DenseMixerConfig':
        """Create configuration from environment variables."""
        config = cls()
        
        # Global settings
        config.enabled = cls._get_env_bool("DENSEMIXER_ENABLED", False)
        config.debug_mode = cls._get_env_bool("DENSEMIXER_DEBUG", False)
        config.log_level = os.environ.get("DENSEMIXER_LOG_LEVEL", "INFO")
        
        # Model-specific settings
        for model_name in config.models:
            env_var = f"DENSEMIXER_{model_name.upper()}"
            config.models[model_name].enabled = cls._get_env_bool(env_var, True)
        
        # Memory settings
        config.memory.use_gradient_checkpointing = cls._get_env_bool("DENSEMIXER_GRADIENT_CHECKPOINT", True)
        config.memory.max_memory_gb = float(os.environ.get("DENSEMIXER_MAX_MEMORY_GB", "16.0"))
        config.memory.memory_efficient_routing = cls._get_env_bool("DENSEMIXER_MEMORY_EFFICIENT", True)
        
        # Adaptive routing settings
        config.adaptive_routing.enabled = cls._get_env_bool("DENSEMIXER_ADAPTIVE_ROUTING", False)
        config.adaptive_routing.min_k = int(os.environ.get("DENSEMIXER_MIN_K", "1"))
        config.adaptive_routing.max_k = int(os.environ.get("DENSEMIXER_MAX_K", "4"))
        
        # Load balancing settings
        config.load_balancing.enabled = cls._get_env_bool("DENSEMIXER_LOAD_BALANCING", True)
        config.load_balancing.balance_factor = float(os.environ.get("DENSEMIXER_BALANCE_FACTOR", "0.01"))
        
        # Monitoring settings
        config.monitoring.enabled = cls._get_env_bool("DENSEMIXER_MONITORING", True)
        config.monitoring.log_interval = int(os.environ.get("DENSEMIXER_LOG_INTERVAL", "100"))
        config.monitoring.save_metrics = cls._get_env_bool("DENSEMIXER_SAVE_METRICS", True)
        
        # Experimental features
        config.experimental.dynamic_expert_selection = cls._get_env_bool("DENSEMIXER_DYNAMIC_EXPERTS", False)
        config.experimental.expert_merging = cls._get_env_bool("DENSEMIXER_EXPERT_MERGING", False)
        
        return config
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> 'DenseMixerConfig':
        """Load configuration from file (JSON or YAML)."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                config_dict = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                config_dict = json.load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
        
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def to_file(self, config_path: Union[str, Path], format: str = 'auto'):
        """Save configuration to file."""
        config_path = Path(config_path)
        config_dict = self.to_dict()
        
        if format == 'auto':
            format = 'yaml' if config_path.suffix.lower() in ['.yaml', '.yml'] else 'json'
        
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            if format == 'yaml':
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            else:
                json.dump(config_dict, f, indent=2)
    
    def is_model_enabled(self, model_name: str) -> bool:
        """Check if a specific model is enabled."""
        if not self.enabled:
            return False
        
        model_config = self.models.get(model_name)
        return model_config is not None and model_config.enabled
    
    def get_model_config(self, model_name: str) -> Optional[ModelSpecificConfig]:
        """Get configuration for a specific model."""
        return self.models.get(model_name)
    
    def update_from_dict(self, updates: Dict[str, Any]):
        """Update configuration with new values."""
        for key, value in updates.items():
            if hasattr(self, key):
                if isinstance(getattr(self, key), dict) and isinstance(value, dict):
                    getattr(self, key).update(value)
                else:
                    setattr(self, key, value)
            else:
                logger.warning(f"Unknown configuration key: {key}")
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        # Validate memory configuration
        if self.memory.max_memory_gb <= 0:
            errors.append("Memory max_memory_gb must be positive")
        
        if self.memory.chunk_size is not None and self.memory.chunk_size <= 0:
            errors.append("Memory chunk_size must be positive")
        
        # Validate adaptive routing configuration
        if self.adaptive_routing.enabled:
            if self.adaptive_routing.min_k >= self.adaptive_routing.max_k:
                errors.append("Adaptive routing min_k must be less than max_k")
            
            if self.adaptive_routing.min_k < 1:
                errors.append("Adaptive routing min_k must be at least 1")
            
            if sum(self.adaptive_routing.complexity_weights.values()) <= 0:
                errors.append("Adaptive routing complexity weights must sum to positive value")
        
        # Validate load balancing configuration
        if self.load_balancing.balance_factor < 0:
            errors.append("Load balancing balance_factor must be non-negative")
        
        if not 0 <= self.load_balancing.expert_dropout_rate < 1:
            errors.append("Load balancing expert_dropout_rate must be in [0, 1)")
        
        # Validate monitoring configuration
        if self.monitoring.history_size <= 0:
            errors.append("Monitoring history_size must be positive")
        
        if self.monitoring.log_interval <= 0:
            errors.append("Monitoring log_interval must be positive")
        
        return errors
    
    @staticmethod
    def _get_env_bool(name: str, default: bool = False) -> bool:
        """Get boolean from environment variable."""
        val = os.environ.get(name, str(default).lower())
        return val.lower() in ("1", "true", "yes", "on", "t")


class ConfigManager:
    """Manages DenseMixer configuration with auto-tuning and persistence."""
    
    def __init__(self, config: Optional[DenseMixerConfig] = None):
        """
        Initialize configuration manager.
        
        Args:
            config: Optional initial configuration
        """
        self.config = config or DenseMixerConfig()
        self.auto_tune_metrics = []
        self.best_config = None
        self.best_score = float('-inf')
        
    def load_config(
        self,
        config_path: Optional[Union[str, Path]] = None,
        from_env: bool = True
    ) -> DenseMixerConfig:
        """
        Load configuration from multiple sources.
        
        Args:
            config_path: Optional path to configuration file
            from_env: Whether to load from environment variables
            
        Returns:
            Loaded configuration
        """
        # Start with default configuration
        config = DenseMixerConfig()
        
        # Load from file if provided
        if config_path is not None:
            try:
                file_config = DenseMixerConfig.from_file(config_path)
                config = file_config
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load configuration from {config_path}: {e}")
        
        # Override with environment variables if requested
        if from_env:
            env_config = DenseMixerConfig.from_env()
            # Merge environment config into loaded config
            config.enabled = env_config.enabled
            config.debug_mode = env_config.debug_mode
            config.log_level = env_config.log_level
            
            # Merge model configurations
            for model_name, env_model_config in env_config.models.items():
                if model_name in config.models:
                    config.models[model_name].enabled = env_model_config.enabled
        
        # Validate configuration
        errors = config.validate()
        if errors:
            logger.error(f"Configuration validation errors: {errors}")
            raise ValueError(f"Invalid configuration: {'; '.join(errors)}")
        
        self.config = config
        return config
    
    def save_config(self, config_path: Union[str, Path], format: str = 'yaml'):
        """Save current configuration to file."""
        self.config.to_file(config_path, format)
        logger.info(f"Saved configuration to {config_path}")
    
    def auto_tune_step(self, metrics: Dict[str, float]):
        """
        Perform one step of auto-tuning based on performance metrics.
        
        Args:
            metrics: Performance metrics from current configuration
        """
        if not self.config.auto_tune:
            return
        
        # Compute overall score (customize based on your objectives)
        score = metrics.get('performance_score', 0.0)
        
        self.auto_tune_metrics.append({
            'config': self.config.to_dict(),
            'metrics': metrics,
            'score': score
        })
        
        # Update best configuration if this is better
        if score > self.best_score:
            self.best_score = score
            self.best_config = DenseMixerConfig.from_dict(self.config.to_dict())
            logger.info(f"New best configuration found with score: {score:.4f}")
        
        # Perform parameter adjustment if we have enough data
        if len(self.auto_tune_metrics) >= 10:
            self._adjust_parameters()
    
    def _adjust_parameters(self):
        """Adjust configuration parameters based on auto-tuning results."""
        # Simple parameter adjustment strategy
        recent_metrics = self.auto_tune_metrics[-10:]
        
        # Analyze load balancing performance
        avg_balance = np.mean([m['metrics'].get('load_balance_coefficient', 0) for m in recent_metrics])
        if avg_balance < 0.8:  # Poor load balancing
            self.config.load_balancing.balance_factor *= 1.1
            logger.info("Increased load balancing factor due to poor balance")
        elif avg_balance > 0.95:  # Excellent load balancing
            self.config.load_balancing.balance_factor *= 0.9
            logger.info("Decreased load balancing factor due to excellent balance")
        
        # Analyze memory usage
        avg_memory = np.mean([m['metrics'].get('memory_usage_gb', 0) for m in recent_metrics])
        if avg_memory > self.config.memory.max_memory_gb * 0.9:  # High memory usage
            if not self.config.memory.use_gradient_checkpointing:
                self.config.memory.use_gradient_checkpointing = True
                logger.info("Enabled gradient checkpointing due to high memory usage")
        
        # Analyze routing performance
        avg_entropy = np.mean([m['metrics'].get('routing_entropy', 0) for m in recent_metrics])
        if avg_entropy < 2.0:  # Low entropy might indicate poor routing
            if not self.config.adaptive_routing.enabled:
                self.config.adaptive_routing.enabled = True
                logger.info("Enabled adaptive routing due to low routing entropy")
    
    def finalize_auto_tune(self) -> Optional[DenseMixerConfig]:
        """
        Finalize auto-tuning and return the best configuration.
        
        Returns:
            Best configuration found during auto-tuning
        """
        if self.best_config is None:
            logger.warning("No best configuration found during auto-tuning")
            return None
        
        logger.info(f"Auto-tuning completed. Best score: {self.best_score:.4f}")
        
        if self.config.save_auto_tune_config:
            # Save best configuration
            best_config_path = Path(self.config.monitoring.metrics_output_dir) / "best_auto_tune_config.yaml"
            self.best_config.to_file(best_config_path)
            logger.info(f"Saved best auto-tuned configuration to {best_config_path}")
        
        return self.best_config
    
    def get_current_config(self) -> DenseMixerConfig:
        """Get current configuration."""
        return self.config
    
    def reset_to_defaults(self):
        """Reset configuration to defaults."""
        self.config = DenseMixerConfig()
        logger.info("Reset configuration to defaults")


# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None


def get_config() -> DenseMixerConfig:
    """Get global DenseMixer configuration."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
        _config_manager.load_config()
    return _config_manager.get_current_config()


def set_config(config: DenseMixerConfig):
    """Set global DenseMixer configuration."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    _config_manager.config = config


def load_config_from_file(config_path: Union[str, Path]) -> DenseMixerConfig:
    """Load configuration from file and set as global config."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager.load_config(config_path)


def create_default_config_file(config_path: Union[str, Path]):
    """Create a default configuration file."""
    config = DenseMixerConfig()
    config.to_file(config_path)
    logger.info(f"Created default configuration file at {config_path}")


# Import numpy for auto-tuning
try:
    import numpy as np
except ImportError:
    logger.warning("NumPy not available for auto-tuning functionality")
    np = None