"""
RLHF Training implementation with contract integration.

Provides PPO training with contractual constraints and multi-stakeholder
reward optimization for legally-compliant AI alignment.
"""

import time
import math
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass
import logging

import jax
import jax.numpy as jnp
from jax import jit, vmap, grad, value_and_grad
import optax
import chex

from ..models.reward_contract import RewardContract
from ..models.legal_blocks import LegalBlocks
from ..utils.error_handling import handle_error, ErrorCategory, ErrorSeverity


@dataclass
class TrainingConfig:
    """Configuration for RLHF training."""
    learning_rate: float = 3e-4
    batch_size: int = 64
    num_epochs: int = 10
    num_steps_per_epoch: int = 2048
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coeff: float = 0.5
    entropy_coeff: float = 0.01
    contract_compliance_coeff: float = 0.1
    max_grad_norm: float = 1.0
    use_advantage_normalization: bool = True
    use_contract_penalties: bool = True
    log_interval: int = 100


@dataclass
class TrainingMetrics:
    """Training metrics and statistics."""
    epoch: int = 0
    step: int = 0
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy_loss: float = 0.0
    contract_penalty: float = 0.0
    total_loss: float = 0.0
    reward_mean: float = 0.0
    reward_std: float = 0.0
    advantage_mean: float = 0.0
    advantage_std: float = 0.0
    contract_violations: int = 0
    gradient_norm: float = 0.0
    learning_rate: float = 0.0
    timestamp: float = 0.0


class ContractualPPO:
    """
    Proximal Policy Optimization with contract constraints.
    
    Integrates RLHF reward contracts into PPO training to ensure
    legal compliance and multi-stakeholder alignment.
    """
    
    def __init__(
        self,
        policy_network: Callable,
        value_network: Callable,
        contract: RewardContract,
        config: Optional[TrainingConfig] = None,
        optimizer: Optional[optax.OptState] = None
    ):
        """Initialize contractual PPO trainer."""
        self.policy_network = policy_network
        self.value_network = value_network
        self.contract = contract
        self.config = config or TrainingConfig()
        
        # Initialize optimizer
        if optimizer is None:
            self.optimizer = optax.chain(
                optax.clip_by_global_norm(self.config.max_grad_norm),
                optax.adam(self.config.learning_rate)
            )
        else:
            self.optimizer = optimizer
        
        # Initialize network parameters
        self.policy_params = None
        self.value_params = None
        self.opt_state = None
        
        # Training state
        self.training_metrics: List[TrainingMetrics] = []
        self.current_epoch = 0
        self.current_step = 0
        self.logger = logging.getLogger(__name__)
        
        # Compile JAX functions for performance
        self._compile_training_functions()
    
    def _compile_training_functions(self):
        """Compile JAX functions for efficient training."""
        # Compile policy and value functions
        self._policy_fn = jit(self._policy_forward)
        self._value_fn = jit(self._value_forward)
        
        # Compile loss functions
        self._policy_loss_fn = jit(self._compute_policy_loss)
        self._value_loss_fn = jit(self._compute_value_loss)
        
        # Compile training step
        self._train_step = jit(self._training_step)
    
    def _policy_forward(self, params: chex.ArrayTree, states: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through policy network."""
        return self.policy_network(params, states)
    
    def _value_forward(self, params: chex.ArrayTree, states: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through value network."""
        return self.value_network(params, states)
    
    def _compute_policy_loss(
        self,
        policy_params: chex.ArrayTree,
        states: jnp.ndarray,
        actions: jnp.ndarray,
        advantages: jnp.ndarray,
        old_log_probs: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute PPO policy loss with clipping."""
        # Get current policy log probabilities
        action_dist = self._policy_fn(policy_params, states)
        log_probs = self._compute_log_probs(action_dist, actions)
        
        # Compute probability ratio
        ratio = jnp.exp(log_probs - old_log_probs)
        
        # Clipped surrogate loss
        clipped_ratio = jnp.clip(
            ratio,
            1 - self.config.clip_epsilon,
            1 + self.config.clip_epsilon
        )
        
        # Policy loss (negative because we want to maximize)
        policy_loss = -jnp.mean(
            jnp.minimum(ratio * advantages, clipped_ratio * advantages)
        )
        
        return policy_loss
    
    def _compute_value_loss(
        self,
        value_params: chex.ArrayTree,
        states: jnp.ndarray,
        returns: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute value function loss."""
        predicted_values = self._value_fn(value_params, states).squeeze()
        value_loss = jnp.mean((predicted_values - returns) ** 2)
        return value_loss
    
    def _compute_entropy_loss(
        self,
        policy_params: chex.ArrayTree,
        states: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute entropy loss for exploration."""
        action_dist = self._policy_fn(policy_params, states)
        entropy = self._compute_entropy(action_dist)
        return -jnp.mean(entropy)  # Negative because we want to maximize entropy
    
    def _compute_contract_penalty(
        self,
        states: jnp.ndarray,
        actions: jnp.ndarray,
        rewards: jnp.ndarray
    ) -> Tuple[jnp.ndarray, int]:
        """Compute penalty for contract violations."""
        total_penalty = 0.0
        violation_count = 0
        
        try:
            # Check contract violations for each state-action pair
            for i in range(len(states)):
                state = states[i]
                action = actions[i]
                
                # Check contract constraints
                violations = self.contract.check_violations(state, action)
                
                if any(violations.values()):
                    violation_count += 1
                    penalty = self.contract.get_violation_penalty(violations)
                    total_penalty += penalty
            
            avg_penalty = total_penalty / len(states) if len(states) > 0 else 0.0
            return jnp.array(avg_penalty), violation_count
            
        except Exception as e:
            self.logger.warning(f"Contract penalty computation failed: {e}")
            return jnp.array(0.0), 0
    
    def _compute_log_probs(
        self,
        action_dist: jnp.ndarray,
        actions: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute log probabilities for actions."""
        # Simplified log probability computation
        # In practice, this would depend on the action distribution type
        log_probs = jnp.sum(actions * jnp.log(action_dist + 1e-8), axis=-1)
        return log_probs
    
    def _compute_entropy(
        self,
        action_dist: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute entropy of action distribution."""
        entropy = -jnp.sum(action_dist * jnp.log(action_dist + 1e-8), axis=-1)
        return entropy
    
    def _training_step(
        self,
        policy_params: chex.ArrayTree,
        value_params: chex.ArrayTree,
        opt_state: optax.OptState,
        batch: Dict[str, jnp.ndarray]
    ) -> Tuple[chex.ArrayTree, chex.ArrayTree, optax.OptState, Dict[str, float]]:
        """Single training step."""
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        advantages = batch['advantages']
        returns = batch['returns']
        old_log_probs = batch['old_log_probs']
        
        # Compute losses
        policy_loss = self._compute_policy_loss(
            policy_params, states, actions, advantages, old_log_probs
        )
        
        value_loss = self._compute_value_loss(
            value_params, states, returns
        )
        
        entropy_loss = self._compute_entropy_loss(
            policy_params, states
        )
        
        # Contract penalty
        contract_penalty, violation_count = self._compute_contract_penalty(
            states, actions, rewards
        )
        
        # Total loss
        total_loss = (
            policy_loss +
            self.config.value_coeff * value_loss +
            self.config.entropy_coeff * entropy_loss +
            self.config.contract_compliance_coeff * contract_penalty
        )
        
        # Compute gradients
        def loss_fn(params):
            policy_p, value_p = params
            p_loss = self._compute_policy_loss(
                policy_p, states, actions, advantages, old_log_probs
            )
            v_loss = self._compute_value_loss(
                value_p, states, returns
            )
            e_loss = self._compute_entropy_loss(policy_p, states)
            return (
                p_loss +
                self.config.value_coeff * v_loss +
                self.config.entropy_coeff * e_loss +
                self.config.contract_compliance_coeff * contract_penalty
            )
        
        loss, grads = value_and_grad(loss_fn)((policy_params, value_params))
        
        # Apply gradients
        updates, new_opt_state = self.optimizer.update(grads, opt_state)
        new_policy_params, new_value_params = optax.apply_updates(
            (policy_params, value_params), updates
        )
        
        # Compute gradient norm
        grad_norm = optax.global_norm(grads)
        
        metrics = {
            'policy_loss': float(policy_loss),
            'value_loss': float(value_loss),
            'entropy_loss': float(entropy_loss),
            'contract_penalty': float(contract_penalty),
            'total_loss': float(total_loss),
            'gradient_norm': float(grad_norm),
            'violation_count': violation_count
        }
        
        return new_policy_params, new_value_params, new_opt_state, metrics
    
    def collect_trajectories(
        self,
        env: Any,
        num_steps: int
    ) -> Dict[str, jnp.ndarray]:
        """Collect trajectories using current policy."""
        trajectories = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': []
        }
        
        state = env.reset()
        
        for step in range(num_steps):
            # Get action from policy
            action_dist = self._policy_fn(self.policy_params, state[None, :])
            action = self._sample_action(action_dist[0])
            
            # Get value estimate
            value = self._value_fn(self.value_params, state[None, :])[0, 0]
            
            # Compute log probability
            log_prob = self._compute_log_probs(action_dist[0:1], action[None, :])[0]
            
            # Take environment step
            next_state, env_reward, done, info = env.step(action)
            
            # Compute contract-aware reward
            try:
                contract_reward = self.contract.compute_reward(
                    jnp.array(state), jnp.array(action)
                )
                # Combine environment and contract rewards
                total_reward = 0.7 * env_reward + 0.3 * contract_reward
            except Exception as e:
                self.logger.warning(f"Contract reward computation failed: {e}")
                total_reward = env_reward
            
            # Store trajectory data
            trajectories['states'].append(state)
            trajectories['actions'].append(action)
            trajectories['rewards'].append(total_reward)
            trajectories['values'].append(value)
            trajectories['log_probs'].append(log_prob)
            trajectories['dones'].append(done)
            
            state = next_state
            
            if done:
                state = env.reset()
        
        # Convert to JAX arrays
        for key in trajectories:
            trajectories[key] = jnp.array(trajectories[key])
        
        return trajectories
    
    def _sample_action(self, action_dist: jnp.ndarray) -> jnp.ndarray:
        """Sample action from distribution."""
        # Simple sampling - in practice would use proper sampling methods
        key = jax.random.PRNGKey(int(time.time() * 1000) % 2**32)
        return jax.random.categorical(key, jnp.log(action_dist + 1e-8))
    
    def compute_advantages(
        self,
        rewards: jnp.ndarray,
        values: jnp.ndarray,
        dones: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute advantages using GAE."""
        advantages = jnp.zeros_like(rewards)
        returns = jnp.zeros_like(rewards)
        
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
                next_non_terminal = 0
            else:
                next_value = values[t + 1]
                next_non_terminal = 1 - dones[t + 1]
            
            delta = rewards[t] + self.config.gamma * next_value * next_non_terminal - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * next_non_terminal * gae
            advantages = advantages.at[t].set(gae)
        
        returns = advantages + values
        
        # Normalize advantages
        if self.config.use_advantage_normalization:
            advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)
        
        return advantages, returns
    
    def train_epoch(
        self,
        env: Any
    ) -> TrainingMetrics:
        """Train for one epoch."""
        epoch_start_time = time.time()
        epoch_metrics = []
        
        for step in range(self.config.num_steps_per_epoch // self.config.batch_size):
            # Collect trajectories
            trajectories = self.collect_trajectories(env, self.config.batch_size)
            
            # Compute advantages and returns
            advantages, returns = self.compute_advantages(
                trajectories['rewards'],
                trajectories['values'],
                trajectories['dones']
            )
            
            # Prepare batch
            batch = {
                'states': trajectories['states'],
                'actions': trajectories['actions'],
                'rewards': trajectories['rewards'],
                'advantages': advantages,
                'returns': returns,
                'old_log_probs': trajectories['log_probs']
            }
            
            # Training step
            (
                self.policy_params,
                self.value_params,
                self.opt_state,
                step_metrics
            ) = self._train_step(
                self.policy_params,
                self.value_params,
                self.opt_state,
                batch
            )
            
            # Update step counter
            self.current_step += 1
            
            # Create training metrics
            metrics = TrainingMetrics(
                epoch=self.current_epoch,
                step=self.current_step,
                policy_loss=step_metrics['policy_loss'],
                value_loss=step_metrics['value_loss'],
                entropy_loss=step_metrics['entropy_loss'],
                contract_penalty=step_metrics['contract_penalty'],
                total_loss=step_metrics['total_loss'],
                reward_mean=float(jnp.mean(trajectories['rewards'])),
                reward_std=float(jnp.std(trajectories['rewards'])),
                advantage_mean=float(jnp.mean(advantages)),
                advantage_std=float(jnp.std(advantages)),
                contract_violations=step_metrics['violation_count'],
                gradient_norm=step_metrics['gradient_norm'],
                learning_rate=self.config.learning_rate,
                timestamp=time.time()
            )
            
            epoch_metrics.append(metrics)
            
            # Log progress
            if step % self.config.log_interval == 0:
                self.logger.info(
                    f"Epoch {self.current_epoch}, Step {step}: "
                    f"Loss={metrics.total_loss:.4f}, "
                    f"Reward={metrics.reward_mean:.4f}, "
                    f"Violations={metrics.contract_violations}"
                )
        
        # Update epoch counter
        self.current_epoch += 1
        
        # Return average metrics for the epoch
        avg_metrics = TrainingMetrics(
            epoch=self.current_epoch - 1,
            step=self.current_step,
            policy_loss=sum(m.policy_loss for m in epoch_metrics) / len(epoch_metrics),
            value_loss=sum(m.value_loss for m in epoch_metrics) / len(epoch_metrics),
            entropy_loss=sum(m.entropy_loss for m in epoch_metrics) / len(epoch_metrics),
            contract_penalty=sum(m.contract_penalty for m in epoch_metrics) / len(epoch_metrics),
            total_loss=sum(m.total_loss for m in epoch_metrics) / len(epoch_metrics),
            reward_mean=sum(m.reward_mean for m in epoch_metrics) / len(epoch_metrics),
            reward_std=sum(m.reward_std for m in epoch_metrics) / len(epoch_metrics),
            advantage_mean=sum(m.advantage_mean for m in epoch_metrics) / len(epoch_metrics),
            advantage_std=sum(m.advantage_std for m in epoch_metrics) / len(epoch_metrics),
            contract_violations=sum(m.contract_violations for m in epoch_metrics),
            gradient_norm=sum(m.gradient_norm for m in epoch_metrics) / len(epoch_metrics),
            learning_rate=self.config.learning_rate,
            timestamp=time.time()
        )
        
        self.training_metrics.append(avg_metrics)
        return avg_metrics
    
    def train(
        self,
        env: Any,
        num_epochs: Optional[int] = None
    ) -> List[TrainingMetrics]:
        """Train the policy for multiple epochs."""
        num_epochs = num_epochs or self.config.num_epochs
        
        self.logger.info(
            f"Starting RLHF training with contract '{self.contract.metadata.name}' "
            f"for {num_epochs} epochs"
        )
        
        for epoch in range(num_epochs):
            try:
                metrics = self.train_epoch(env)
                
                # Log epoch summary
                self.logger.info(
                    f"Epoch {epoch} completed: "
                    f"Avg Loss={metrics.total_loss:.4f}, "
                    f"Avg Reward={metrics.reward_mean:.4f}, "
                    f"Total Violations={metrics.contract_violations}"
                )
                
                # Check for contract compliance
                if metrics.contract_violations > 0:
                    self.logger.warning(
                        f"Contract violations detected in epoch {epoch}: "
                        f"{metrics.contract_violations} violations"
                    )
                
            except Exception as e:
                handle_error(
                    error=e,
                    operation=f"train_epoch:{epoch}",
                    category=ErrorCategory.TRAINING,
                    severity=ErrorSeverity.HIGH,
                    additional_info={
                        "contract_name": self.contract.metadata.name,
                        "epoch": epoch
                    }
                )
                self.logger.error(f"Training epoch {epoch} failed: {e}")
                break
        
        self.logger.info("RLHF training completed")
        return self.training_metrics
    
    def initialize_networks(
        self,
        state_dim: int,
        action_dim: int,
        key: jax.random.PRNGKey
    ):
        """Initialize network parameters."""
        # Initialize policy parameters
        dummy_state = jnp.zeros((1, state_dim))
        self.policy_params = self.policy_network.init(key, dummy_state)
        
        # Initialize value parameters  
        key, subkey = jax.random.split(key)
        self.value_params = self.value_network.init(subkey, dummy_state)
        
        # Initialize optimizer state
        combined_params = (self.policy_params, self.value_params)
        self.opt_state = self.optimizer.init(combined_params)
    
    def save_checkpoint(
        self,
        filepath: str
    ):
        """Save training checkpoint."""
        import pickle
        
        checkpoint = {
            'policy_params': self.policy_params,
            'value_params': self.value_params,
            'opt_state': self.opt_state,
            'current_epoch': self.current_epoch,
            'current_step': self.current_step,
            'training_metrics': self.training_metrics,
            'config': self.config,
            'contract_hash': self.contract.compute_hash()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint, f)
    
    def load_checkpoint(
        self,
        filepath: str
    ):
        """Load training checkpoint."""
        import pickle
        
        with open(filepath, 'rb') as f:
            checkpoint = pickle.load(f)
        
        self.policy_params = checkpoint['policy_params']
        self.value_params = checkpoint['value_params']
        self.opt_state = checkpoint['opt_state']
        self.current_epoch = checkpoint['current_epoch']
        self.current_step = checkpoint['current_step']
        self.training_metrics = checkpoint['training_metrics']
        
        # Verify contract compatibility
        if checkpoint['contract_hash'] != self.contract.compute_hash():
            self.logger.warning(
                "Contract hash mismatch - checkpoint may be incompatible"
            )
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training progress."""
        if not self.training_metrics:
            return {"status": "not_started"}
        
        latest_metrics = self.training_metrics[-1]
        
        return {
            "status": "completed" if self.current_epoch >= self.config.num_epochs else "in_progress",
            "current_epoch": self.current_epoch,
            "total_epochs": self.config.num_epochs,
            "current_step": self.current_step,
            "latest_metrics": {
                "total_loss": latest_metrics.total_loss,
                "policy_loss": latest_metrics.policy_loss,
                "value_loss": latest_metrics.value_loss,
                "reward_mean": latest_metrics.reward_mean,
                "contract_violations": latest_metrics.contract_violations
            },
            "contract_name": self.contract.metadata.name,
            "contract_hash": self.contract.compute_hash(),
            "total_training_time": (
                latest_metrics.timestamp - self.training_metrics[0].timestamp
                if len(self.training_metrics) > 0 else 0.0
            )
        }
