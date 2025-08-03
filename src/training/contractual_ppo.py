"""
Contractual PPO implementation.

Implements Proximal Policy Optimization with contract constraint enforcement
for safe and legally compliant RLHF training.
"""

import time
from typing import Dict, List, Optional, Callable, Any, Tuple, NamedTuple
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state

from ..models.reward_contract import RewardContract
from ..models.reward_model import ContractualRewardModel


@dataclass
class PPOConfig:
    """Configuration for Contractual PPO."""
    learning_rate: float = 3e-4
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    contract_coef: float = 0.1
    gae_lambda: float = 0.95
    gamma: float = 0.99
    max_grad_norm: float = 1.0
    num_epochs: int = 4
    batch_size: int = 256
    rollout_length: int = 2048
    normalize_advantages: bool = True
    use_contract_shaping: bool = True
    violation_penalty_scale: float = 2.0


@dataclass
class PPOMetrics:
    """Training metrics for PPO."""
    step: int
    policy_loss: float
    value_loss: float
    entropy_loss: float
    contract_loss: float
    total_loss: float
    clip_fraction: float
    explained_variance: float
    contract_compliance_rate: float
    mean_reward: float
    mean_episode_length: float
    training_time: float
    violation_breakdown: Dict[str, float]


class Trajectory(NamedTuple):
    """Single trajectory data."""
    observations: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    dones: jnp.ndarray
    log_probs: jnp.ndarray
    values: jnp.ndarray
    advantages: jnp.ndarray
    returns: jnp.ndarray
    contract_violations: Dict[str, jnp.ndarray]


class PolicyNetwork(nn.Module):
    """Policy network for PPO."""
    
    action_dim: int
    hidden_dim: int = 256
    
    def setup(self):
        """Initialize network layers."""
        self.layers = [
            nn.Dense(self.hidden_dim),
            nn.relu,
            nn.Dense(self.hidden_dim),
            nn.relu,
            nn.Dense(self.action_dim)
        ]
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass."""
        for layer in self.layers:
            x = layer(x)
        return x


class ValueNetwork(nn.Module):
    """Value network for PPO."""
    
    hidden_dim: int = 256
    
    def setup(self):
        """Initialize network layers."""
        self.layers = [
            nn.Dense(self.hidden_dim),
            nn.relu,
            nn.Dense(self.hidden_dim), 
            nn.relu,
            nn.Dense(1)
        ]
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass."""
        for layer in self.layers:
            x = layer(x)
        return x.squeeze(-1)


class ContractualPPO:
    """
    Contractual Proximal Policy Optimization.
    
    Implements PPO with contract constraint enforcement for safe RLHF training.
    Ensures that policy updates respect legal and safety constraints encoded
    in the reward contract.
    """
    
    def __init__(
        self,
        policy_network: PolicyNetwork,
        value_network: ValueNetwork,
        contract: RewardContract,
        reward_model: ContractualRewardModel,
        config: PPOConfig,
        random_key: jax.random.PRNGKey
    ):
        """
        Initialize Contractual PPO.
        
        Args:
            policy_network: Policy network
            value_network: Value network  
            contract: Reward contract to enforce
            reward_model: Contractual reward model
            config: PPO configuration
            random_key: JAX random key
        """
        self.config = config
        self.contract = contract
        self.reward_model = reward_model
        
        # Initialize networks
        self.policy_net = policy_network
        self.value_net = value_network
        
        # Initialize optimizers
        self.policy_optimizer = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(config.learning_rate)
        )
        self.value_optimizer = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(config.learning_rate)
        )
        
        # Initialize parameters
        key1, key2 = jax.random.split(random_key)
        dummy_obs = jnp.ones((1, 128))  # Dummy observation
        
        self.policy_params = policy_network.init(key1, dummy_obs)
        self.value_params = value_network.init(key2, dummy_obs)
        
        # Initialize optimizer states
        self.policy_opt_state = self.policy_optimizer.init(self.policy_params)
        self.value_opt_state = self.value_optimizer.init(self.value_params)
        
        # Training state
        self.step = 0
        self.training_metrics: List[PPOMetrics] = []
        
        # JIT compile functions for performance
        self._jit_policy_loss = jax.jit(self._compute_policy_loss)
        self._jit_value_loss = jax.jit(self._compute_value_loss)
        self._jit_policy_update = jax.jit(self._update_policy)
        self._jit_value_update = jax.jit(self._update_value)
    
    def collect_trajectories(
        self,
        env: Any,
        num_steps: int = None
    ) -> List[Trajectory]:
        """
        Collect trajectories from environment with contract monitoring.
        
        Args:
            env: Environment to collect from
            num_steps: Number of steps to collect (uses config default if None)
            
        Returns:
            List of collected trajectories
        """
        if num_steps is None:
            num_steps = self.config.rollout_length
        
        trajectories = []
        observations = []
        actions = []
        rewards = []
        dones = []
        log_probs = []
        values = []
        contract_violations = {name: [] for name in self.contract.constraints.keys()}
        
        obs = env.reset()
        
        for step in range(num_steps):
            # Get action from policy
            action, log_prob = self._sample_action(obs)
            value = self._compute_value(obs)
            
            # Take environment step
            next_obs, base_reward, done, info = env.step(action)
            
            # Apply contract constraints to reward
            contract_reward, violations = self._apply_contract_to_reward(
                obs, action, base_reward, info
            )
            
            # Store trajectory data
            observations.append(obs)
            actions.append(action)
            rewards.append(contract_reward)
            dones.append(done)
            log_probs.append(log_prob)
            values.append(value)
            
            # Store violation data
            for constraint_name, violated in violations.items():
                contract_violations[constraint_name].append(violated)
            
            obs = next_obs
            
            if done:
                # Compute advantages and returns
                advantages, returns = self._compute_gae(
                    jnp.array(rewards),
                    jnp.array(values),
                    jnp.array(dones)
                )
                
                # Create trajectory
                trajectory = Trajectory(
                    observations=jnp.array(observations),
                    actions=jnp.array(actions),
                    rewards=jnp.array(rewards),
                    dones=jnp.array(dones),
                    log_probs=jnp.array(log_probs),
                    values=jnp.array(values),
                    advantages=advantages,
                    returns=returns,
                    contract_violations={
                        name: jnp.array(viols)
                        for name, viols in contract_violations.items()
                    }
                )
                trajectories.append(trajectory)
                
                # Reset for next trajectory
                observations.clear()
                actions.clear()
                rewards.clear()
                dones.clear()
                log_probs.clear()
                values.clear()
                contract_violations = {name: [] for name in self.contract.constraints.keys()}
                
                obs = env.reset()
        
        return trajectories
    
    def _sample_action(self, observation: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Sample action from policy."""
        logits = self.policy_net.apply(self.policy_params, observation[None])
        action_dist = jax.nn.softmax(logits)
        
        # Sample action
        key = jax.random.PRNGKey(int(time.time() * 1000000) % 2**32)
        action = jax.random.categorical(key, logits)
        log_prob = jax.nn.log_softmax(logits)[0, action]
        
        return action, log_prob
    
    def _compute_value(self, observation: jnp.ndarray) -> jnp.ndarray:
        """Compute state value."""
        return self.value_net.apply(self.value_params, observation[None])[0]
    
    def _apply_contract_to_reward(
        self,
        obs: jnp.ndarray,
        action: jnp.ndarray,
        base_reward: float,
        info: Dict[str, Any]
    ) -> Tuple[float, Dict[str, bool]]:
        """Apply contract constraints to base reward."""
        contract_reward = base_reward
        violations = {}
        
        # Check each constraint
        for constraint_name, constraint in self.contract.constraints.items():
            try:
                # Create mock state and action for constraint checking
                state = type('State', (), {
                    'observation': obs,
                    'action': action,
                    'info': info
                })()
                
                action_obj = type('Action', (), {
                    'value': action,
                    'output': f"action_{action}",
                    'reward': base_reward
                })()
                
                # Check constraint
                constraint_satisfied = constraint.constraint_fn(state, action_obj)
                
                if not constraint_satisfied:
                    # Apply penalty
                    penalty = constraint.violation_penalty * constraint.severity
                    contract_reward += penalty
                    violations[constraint_name] = True
                else:
                    violations[constraint_name] = False
                    
            except Exception:
                # Constraint evaluation failed, apply penalty
                penalty = constraint.violation_penalty * constraint.severity  
                contract_reward += penalty
                violations[constraint_name] = True
        
        return contract_reward, violations
    
    def _compute_gae(
        self,
        rewards: jnp.ndarray,
        values: jnp.ndarray,
        dones: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute Generalized Advantage Estimation."""
        advantages = jnp.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.config.gamma * next_value * (1 - dones[t]) - values[t]
            advantages = advantages.at[t].set(
                delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * last_advantage
            )
            last_advantage = advantages[t]
        
        returns = advantages + values
        
        if self.config.normalize_advantages:
            advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)
        
        return advantages, returns
    
    def update(self, trajectories: List[Trajectory]) -> PPOMetrics:
        """
        Update policy and value networks using collected trajectories.
        
        Args:
            trajectories: Collected trajectory data
            
        Returns:
            Training metrics
        """
        start_time = time.time()
        
        # Combine all trajectories
        combined_data = self._combine_trajectories(trajectories)
        
        # Perform multiple epochs of updates
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        total_contract_loss = 0
        clip_fractions = []
        
        for epoch in range(self.config.num_epochs):
            # Shuffle data
            batch_indices = jax.random.permutation(
                jax.random.PRNGKey(epoch), 
                len(combined_data['observations'])
            )
            
            # Process in batches
            for i in range(0, len(batch_indices), self.config.batch_size):
                batch_idx = batch_indices[i:i + self.config.batch_size]
                batch_data = {
                    key: value[batch_idx] for key, value in combined_data.items()
                    if isinstance(value, jnp.ndarray)
                }
                
                # Update policy
                policy_loss, policy_metrics = self._jit_policy_update(
                    self.policy_params,
                    self.policy_opt_state,
                    batch_data
                )
                
                # Update value function
                value_loss, value_metrics = self._jit_value_update(
                    self.value_params,
                    self.value_opt_state,
                    batch_data
                )
                
                total_policy_loss += policy_loss
                total_value_loss += value_loss
                total_entropy_loss += policy_metrics.get('entropy_loss', 0)
                total_contract_loss += policy_metrics.get('contract_loss', 0)
                clip_fractions.append(policy_metrics.get('clip_fraction', 0))
        
        # Compute training metrics
        num_batches = (len(combined_data['observations']) // self.config.batch_size) * self.config.num_epochs
        
        metrics = PPOMetrics(
            step=self.step,
            policy_loss=total_policy_loss / num_batches,
            value_loss=total_value_loss / num_batches,
            entropy_loss=total_entropy_loss / num_batches,
            contract_loss=total_contract_loss / num_batches,
            total_loss=(total_policy_loss + total_value_loss) / num_batches,
            clip_fraction=jnp.mean(jnp.array(clip_fractions)),
            explained_variance=self._compute_explained_variance(combined_data),
            contract_compliance_rate=self._compute_compliance_rate(trajectories),
            mean_reward=jnp.mean(combined_data['rewards']),
            mean_episode_length=len(combined_data['observations']) / len(trajectories),
            training_time=time.time() - start_time,
            violation_breakdown=self._compute_violation_breakdown(trajectories)
        )
        
        self.step += 1
        self.training_metrics.append(metrics)
        
        return metrics
    
    def _combine_trajectories(self, trajectories: List[Trajectory]) -> Dict[str, jnp.ndarray]:
        """Combine multiple trajectories into single arrays."""
        combined = {}
        
        # Standard trajectory fields
        for field in ['observations', 'actions', 'rewards', 'dones', 'log_probs', 'values', 'advantages', 'returns']:
            combined[field] = jnp.concatenate([getattr(traj, field) for traj in trajectories])
        
        # Contract violation fields
        if trajectories:
            for constraint_name in trajectories[0].contract_violations.keys():
                combined[f'violations_{constraint_name}'] = jnp.concatenate([
                    traj.contract_violations[constraint_name] for traj in trajectories
                ])
        
        return combined
    
    def _compute_policy_loss(
        self,
        params: Dict,
        batch_data: Dict[str, jnp.ndarray]
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """Compute policy loss with contract constraints."""
        # Get current policy predictions
        logits = self.policy_net.apply(params, batch_data['observations'])
        log_probs = jax.nn.log_softmax(logits)
        current_log_probs = log_probs[jnp.arange(len(batch_data['actions'])), batch_data['actions']]
        
        # Compute ratio
        ratio = jnp.exp(current_log_probs - batch_data['log_probs'])
        
        # Clipped surrogate loss
        advantages = batch_data['advantages']
        surr1 = ratio * advantages
        surr2 = jnp.clip(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantages
        policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))
        
        # Entropy bonus
        entropy = -jnp.sum(jax.nn.softmax(logits) * log_probs, axis=1)
        entropy_loss = -self.config.entropy_coef * jnp.mean(entropy)
        
        # Contract constraint loss
        contract_loss = self._compute_contract_policy_loss(batch_data)
        
        total_loss = policy_loss + entropy_loss + self.config.contract_coef * contract_loss
        
        # Metrics
        clip_fraction = jnp.mean((jnp.abs(ratio - 1.0) > self.config.clip_epsilon).astype(jnp.float32))
        
        return total_loss, {
            'policy_loss': policy_loss,
            'entropy_loss': entropy_loss,
            'contract_loss': contract_loss,
            'clip_fraction': clip_fraction
        }
    
    def _compute_value_loss(
        self,
        params: Dict,
        batch_data: Dict[str, jnp.ndarray]
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """Compute value function loss."""
        predicted_values = self.value_net.apply(params, batch_data['observations'])
        value_loss = jnp.mean((predicted_values - batch_data['returns']) ** 2)
        
        return value_loss, {'value_loss': value_loss}
    
    def _compute_contract_policy_loss(self, batch_data: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """Compute contract-specific policy loss."""
        contract_loss = 0.0
        
        # Add penalty for actions that led to violations
        for constraint_name in self.contract.constraints.keys():
            violation_key = f'violations_{constraint_name}'
            if violation_key in batch_data:
                violations = batch_data[violation_key]
                # Penalize policy for taking actions that violate constraints
                contract_loss += jnp.mean(violations * self.config.violation_penalty_scale)
        
        return contract_loss
    
    def _update_policy(
        self,
        params: Dict,
        opt_state: Any,
        batch_data: Dict[str, jnp.ndarray]
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """Update policy parameters."""
        (loss, metrics), grads = jax.value_and_grad(self._compute_policy_loss, has_aux=True)(
            params, batch_data
        )
        
        updates, new_opt_state = self.policy_optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        
        # Update stored parameters
        self.policy_params = new_params
        self.policy_opt_state = new_opt_state
        
        return loss, metrics
    
    def _update_value(
        self,
        params: Dict,
        opt_state: Any,
        batch_data: Dict[str, jnp.ndarray]
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """Update value function parameters."""
        (loss, metrics), grads = jax.value_and_grad(self._compute_value_loss, has_aux=True)(
            params, batch_data
        )
        
        updates, new_opt_state = self.value_optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        
        # Update stored parameters
        self.value_params = new_params
        self.value_opt_state = new_opt_state
        
        return loss, metrics
    
    def _compute_explained_variance(self, data: Dict[str, jnp.ndarray]) -> float:
        """Compute explained variance of value function."""
        values = data['values']
        returns = data['returns']
        
        var_returns = jnp.var(returns)
        if var_returns == 0:
            return 0.0
        
        return 1.0 - jnp.var(returns - values) / var_returns
    
    def _compute_compliance_rate(self, trajectories: List[Trajectory]) -> float:
        """Compute overall contract compliance rate."""
        total_steps = 0
        total_violations = 0
        
        for trajectory in trajectories:
            for constraint_name, violations in trajectory.contract_violations.items():
                total_steps += len(violations)
                total_violations += jnp.sum(violations)
        
        if total_steps == 0:
            return 1.0
        
        return 1.0 - (total_violations / total_steps)
    
    def _compute_violation_breakdown(self, trajectories: List[Trajectory]) -> Dict[str, float]:
        """Compute violation rates by constraint type."""
        violation_breakdown = {}
        
        for constraint_name in self.contract.constraints.keys():
            total_steps = 0
            total_violations = 0
            
            for trajectory in trajectories:
                if constraint_name in trajectory.contract_violations:
                    violations = trajectory.contract_violations[constraint_name]
                    total_steps += len(violations)
                    total_violations += jnp.sum(violations)
            
            if total_steps > 0:
                violation_breakdown[constraint_name] = total_violations / total_steps
            else:
                violation_breakdown[constraint_name] = 0.0
        
        return violation_breakdown
    
    def apply_penalty(self, violations: Dict[str, List[bool]]) -> None:
        """Apply penalties for contract violations."""
        # Update policy to discourage violation-prone actions
        # This is a simplified implementation
        penalty_scale = 0.1
        
        for constraint_name, violation_list in violations.items():
            if any(violation_list):
                # Increase penalty for this constraint type
                if constraint_name in self.contract.constraints:
                    constraint = self.contract.constraints[constraint_name]
                    constraint.violation_penalty *= (1 + penalty_scale)
    
    def sign_checkpoint(self) -> str:
        """Sign training checkpoint for audit trail."""
        import hashlib
        
        checkpoint_data = {
            'step': self.step,
            'policy_params_hash': str(hash(str(self.policy_params))),
            'value_params_hash': str(hash(str(self.value_params))),
            'contract_hash': self.contract.compute_hash(),
            'timestamp': time.time()
        }
        
        checkpoint_str = str(checkpoint_data)
        signature = hashlib.sha256(checkpoint_str.encode()).hexdigest()
        
        return signature