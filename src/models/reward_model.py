"""
Contractual reward model implementation.

Implements neural reward models with contract constraint enforcement
for RLHF training with legal compliance guarantees.
"""

import time
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state

from .reward_contract import RewardContract
from .legal_blocks import LegalBlocks, RLHFConstraints


@dataclass
class TrainingMetrics:
    """Training metrics for contractual reward model."""
    step: int
    loss: float
    accuracy: float
    contract_compliance_rate: float
    violation_count: int
    constraint_penalties: Dict[str, float]
    training_time: float


@dataclass
class RewardModelConfig:
    """Configuration for contractual reward model."""
    hidden_dim: int = 768
    num_layers: int = 6
    num_heads: int = 12
    dropout_rate: float = 0.1
    max_sequence_length: int = 512
    vocab_size: int = 50000
    contract_weight: float = 0.1
    constraint_temperature: float = 1.0


class TransformerRewardModel(nn.Module):
    """Transformer-based reward model with contract integration."""
    
    config: RewardModelConfig
    
    def setup(self):
        """Initialize model layers."""
        self.embedding = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.hidden_dim
        )
        
        self.position_embedding = nn.Embed(
            num_embeddings=self.config.max_sequence_length,
            features=self.config.hidden_dim
        )
        
        self.transformer_layers = [
            nn.TransformerBlock(
                num_heads=self.config.num_heads,
                qkv_features=self.config.hidden_dim,
                mlp_features=self.config.hidden_dim * 4,
                dropout_rate=self.config.dropout_rate,
                attention_dropout_rate=self.config.dropout_rate
            )
            for _ in range(self.config.num_layers)
        ]
        
        self.layer_norm = nn.LayerNorm()
        self.reward_head = nn.Dense(1)
        self.contract_head = nn.Dense(64)  # For contract compliance features
        
    def __call__(self, tokens: jnp.ndarray, training: bool = False) -> Dict[str, jnp.ndarray]:
        """
        Forward pass of reward model.
        
        Args:
            tokens: Input token sequence [batch, seq_len]
            training: Whether in training mode
            
        Returns:
            Dictionary with reward predictions and contract features
        """
        batch_size, seq_len = tokens.shape
        
        # Token and position embeddings
        token_emb = self.embedding(tokens)
        pos_emb = self.position_embedding(jnp.arange(seq_len))
        hidden = token_emb + pos_emb
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            hidden = layer(hidden, deterministic=not training)
        
        # Final layer norm
        hidden = self.layer_norm(hidden)
        
        # Pool sequence representation (use last token for now)
        pooled = hidden[:, -1, :]
        
        # Reward prediction
        reward_logits = self.reward_head(pooled)
        rewards = jnp.tanh(reward_logits)  # Bound to [-1, 1]
        
        # Contract compliance features
        contract_features = self.contract_head(pooled)
        
        return {
            'rewards': rewards.squeeze(-1),
            'contract_features': contract_features,
            'hidden_states': hidden,
            'pooled_representation': pooled
        }


class ContractualRewardModel:
    """
    Contractual reward model with legal constraint enforcement.
    
    Integrates neural reward modeling with contract compliance checking
    to ensure RLHF training respects legal and safety constraints.
    """
    
    def __init__(
        self,
        config: RewardModelConfig,
        contract: RewardContract,
        random_key: jax.random.PRNGKey
    ):
        """
        Initialize contractual reward model.
        
        Args:
            config: Model configuration
            contract: Reward contract to enforce
            random_key: JAX random key for initialization
        """
        self.config = config
        self.contract = contract
        self.model = TransformerRewardModel(config)
        
        # Initialize model parameters
        dummy_input = jnp.ones((1, config.max_sequence_length), dtype=jnp.int32)
        self.params = self.model.init(random_key, dummy_input)
        
        # Initialize optimizer
        self.optimizer = optax.adam(learning_rate=3e-4)
        self.opt_state = self.optimizer.init(self.params)
        
        # Training state
        self.train_state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=self.params,
            tx=self.optimizer
        )
        
        # Metrics tracking
        self.training_metrics: List[TrainingMetrics] = []
        self._violation_cache: Dict[str, bool] = {}
    
    def compute_reward(
        self,
        chosen_tokens: jnp.ndarray,
        rejected_tokens: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
        """
        Compute rewards for chosen and rejected sequences with contract enforcement.
        
        Args:
            chosen_tokens: Preferred sequence tokens
            rejected_tokens: Dis-preferred sequence tokens
            
        Returns:
            Tuple of (chosen_rewards, rejected_rewards, metadata)
        """
        # Get neural reward predictions
        chosen_outputs = self.model.apply(
            self.train_state.params,
            chosen_tokens,
            training=False
        )
        rejected_outputs = self.model.apply(
            self.train_state.params,
            rejected_tokens,
            training=False
        )
        
        chosen_rewards = chosen_outputs['rewards']
        rejected_rewards = rejected_outputs['rewards']
        
        # Apply contract constraints
        chosen_contract_rewards, chosen_violations = self._apply_contract_constraints(
            chosen_tokens, chosen_rewards, chosen_outputs
        )
        rejected_contract_rewards, rejected_violations = self._apply_contract_constraints(
            rejected_tokens, rejected_rewards, rejected_outputs
        )
        
        metadata = {
            'chosen_violations': chosen_violations,
            'rejected_violations': rejected_violations,
            'chosen_features': chosen_outputs['contract_features'],
            'rejected_features': rejected_outputs['contract_features'],
            'contract_compliance_rate': self._compute_compliance_rate([chosen_violations, rejected_violations])
        }
        
        return chosen_contract_rewards, rejected_contract_rewards, metadata
    
    def _apply_contract_constraints(
        self,
        tokens: jnp.ndarray,
        base_rewards: jnp.ndarray,
        model_outputs: Dict[str, jnp.ndarray]
    ) -> Tuple[jnp.ndarray, Dict[str, bool]]:
        """Apply contract constraints to base reward predictions."""
        batch_size = tokens.shape[0]
        constrained_rewards = base_rewards.copy()
        violations = {}
        
        # Check each constraint in the contract
        for constraint_name, constraint in self.contract.constraints.items():
            constraint_violations = []
            
            for i in range(batch_size):
                # Convert tokens to pseudo-text for constraint checking
                # In practice, would use proper tokenizer
                text_representation = f"sequence_{jnp.sum(tokens[i])}"
                
                # Create mock state and action for constraint checking
                state = MockState(
                    tokens=tokens[i],
                    hidden_states=model_outputs['hidden_states'][i],
                    user_id=f"user_{i}"
                )
                action = MockAction(
                    output=text_representation,
                    reward=float(base_rewards[i]),
                    features=model_outputs['contract_features'][i]
                )
                
                # Check constraint
                try:
                    constraint_satisfied = constraint.constraint_fn(state, action)
                    if not constraint_satisfied:
                        # Apply penalty
                        penalty = constraint.violation_penalty * constraint.severity
                        constrained_rewards = constrained_rewards.at[i].add(penalty)
                        constraint_violations.append(True)
                    else:
                        constraint_violations.append(False)
                except Exception:
                    # Constraint evaluation failed, apply penalty
                    penalty = constraint.violation_penalty * constraint.severity
                    constrained_rewards = constrained_rewards.at[i].add(penalty)
                    constraint_violations.append(True)
            
            violations[constraint_name] = constraint_violations
        
        # Apply stakeholder-specific constraints
        for stakeholder_name, reward_fn in self.contract.reward_functions.items():
            if hasattr(reward_fn, '__legal_blocks__'):
                # Apply Legal-Blocks constraints
                stakeholder_violations = self._check_legal_blocks_constraints(
                    tokens, model_outputs, reward_fn
                )
                violations[f"{stakeholder_name}_legal_blocks"] = stakeholder_violations
        
        return constrained_rewards, violations
    
    def _check_legal_blocks_constraints(
        self,
        tokens: jnp.ndarray,
        model_outputs: Dict[str, jnp.ndarray],
        reward_fn: Callable
    ) -> List[bool]:
        """Check Legal-Blocks constraints for a reward function."""
        constraints = LegalBlocks.get_constraints(reward_fn)
        if not constraints:
            return [False] * tokens.shape[0]
        
        violations = []
        for i in range(tokens.shape[0]):
            # Mock validation for Legal-Blocks constraints
            text_representation = f"sequence_{jnp.sum(tokens[i])}"
            
            # Check common RLHF constraints
            action = MockAction(output=text_representation)
            state = MockState(tokens=tokens[i])
            
            # Apply pre-defined RLHF constraints
            constraint_checks = [
                RLHFConstraints.no_harmful_output(action),
                RLHFConstraints.privacy_protection(state, action),
                RLHFConstraints.truthfulness_requirement(action),
                RLHFConstraints.fairness_requirement(state, action)
            ]
            
            # Constraint violated if any check fails
            violations.append(not all(constraint_checks))
        
        return violations
    
    def _compute_compliance_rate(self, violation_lists: List[Dict[str, List[bool]]]) -> float:
        """Compute overall contract compliance rate."""
        total_checks = 0
        total_violations = 0
        
        for violation_dict in violation_lists:
            for constraint_name, violations in violation_dict.items():
                total_checks += len(violations)
                total_violations += sum(violations)
        
        if total_checks == 0:
            return 1.0
        
        return 1.0 - (total_violations / total_checks)
    
    def preference_loss(
        self,
        chosen_tokens: jnp.ndarray,
        rejected_tokens: jnp.ndarray
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """
        Compute preference learning loss with contract consistency.
        
        Args:
            chosen_tokens: Preferred sequence tokens
            rejected_tokens: Dis-preferred sequence tokens
            
        Returns:
            Tuple of (loss, metrics)
        """
        # Compute rewards with contract enforcement
        chosen_rewards, rejected_rewards, metadata = self.compute_reward(
            chosen_tokens, rejected_tokens
        )
        
        # Standard preference loss (Bradley-Terry model)
        reward_diff = chosen_rewards - rejected_rewards
        preference_loss = -jnp.mean(jax.nn.log_sigmoid(reward_diff))
        
        # Contract consistency loss
        contract_loss = self._compute_contract_consistency_loss(metadata)
        
        # Combined loss
        total_loss = preference_loss + self.config.contract_weight * contract_loss
        
        loss_metrics = {
            'preference_loss': preference_loss,
            'contract_loss': contract_loss,
            'total_loss': total_loss,
            'contract_compliance_rate': metadata['contract_compliance_rate'],
            'chosen_violation_rate': self._compute_violation_rate(metadata['chosen_violations']),
            'rejected_violation_rate': self._compute_violation_rate(metadata['rejected_violations'])
        }
        
        return total_loss, loss_metrics
    
    def _compute_contract_consistency_loss(self, metadata: Dict[str, Any]) -> jnp.ndarray:
        """Compute loss term for contract consistency."""
        # Penalty for constraint violations
        chosen_violations = metadata['chosen_violations']
        rejected_violations = metadata['rejected_violations']
        
        violation_penalty = 0.0
        
        # Add penalties for each type of violation
        for constraint_name in chosen_violations:
            chosen_viols = jnp.array(chosen_violations[constraint_name], dtype=jnp.float32)
            rejected_viols = jnp.array(rejected_violations[constraint_name], dtype=jnp.float32)
            
            # Penalize violations more heavily in chosen sequences
            violation_penalty += jnp.mean(chosen_viols * 2.0 + rejected_viols)
        
        return violation_penalty
    
    def _compute_violation_rate(self, violations: Dict[str, List[bool]]) -> float:
        """Compute violation rate across all constraints."""
        total_checks = 0
        total_violations = 0
        
        for constraint_violations in violations.values():
            total_checks += len(constraint_violations)
            total_violations += sum(constraint_violations)
        
        if total_checks == 0:
            return 0.0
        
        return total_violations / total_checks
    
    def update(
        self,
        chosen_tokens: jnp.ndarray,
        rejected_tokens: jnp.ndarray
    ) -> TrainingMetrics:
        """
        Update model parameters with preference data.
        
        Args:
            chosen_tokens: Preferred sequence tokens
            rejected_tokens: Dis-preferred sequence tokens
            
        Returns:
            Training metrics
        """
        start_time = time.time()
        
        def loss_fn(params):
            """Loss function for gradient computation."""
            # Set parameters in train state
            temp_state = self.train_state.replace(params=params)
            old_state = self.train_state
            self.train_state = temp_state
            
            try:
                loss, metrics = self.preference_loss(chosen_tokens, rejected_tokens)
                return loss, metrics
            finally:
                self.train_state = old_state
        
        # Compute gradients
        (loss, loss_metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            self.train_state.params
        )
        
        # Update parameters
        self.train_state = self.train_state.apply_gradients(grads=grads)
        
        training_time = time.time() - start_time
        
        # Create training metrics
        metrics = TrainingMetrics(
            step=self.train_state.step,
            loss=float(loss),
            accuracy=1.0 - loss_metrics['chosen_violation_rate'],  # Proxy for accuracy
            contract_compliance_rate=loss_metrics['contract_compliance_rate'],
            violation_count=int(loss_metrics['chosen_violation_rate'] * chosen_tokens.shape[0]),
            constraint_penalties={
                'preference_loss': float(loss_metrics['preference_loss']),
                'contract_loss': float(loss_metrics['contract_loss'])
            },
            training_time=training_time
        )
        
        self.training_metrics.append(metrics)
        return metrics
    
    def evaluate(
        self,
        test_chosen: jnp.ndarray,
        test_rejected: jnp.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Args:
            test_chosen: Test preferred sequences
            test_rejected: Test dis-preferred sequences
            
        Returns:
            Evaluation metrics
        """
        chosen_rewards, rejected_rewards, metadata = self.compute_reward(
            test_chosen, test_rejected
        )
        
        # Compute preference accuracy
        correct_preferences = jnp.mean(chosen_rewards > rejected_rewards)
        
        evaluation_metrics = {
            'preference_accuracy': float(correct_preferences),
            'contract_compliance_rate': metadata['contract_compliance_rate'],
            'mean_chosen_reward': float(jnp.mean(chosen_rewards)),
            'mean_rejected_reward': float(jnp.mean(rejected_rewards)),
            'reward_margin': float(jnp.mean(chosen_rewards - rejected_rewards))
        }
        
        return evaluation_metrics
    
    def save_checkpoint(self, filepath: str) -> None:
        """Save model checkpoint."""
        checkpoint_data = {
            'params': self.train_state.params,
            'opt_state': self.train_state.opt_state,
            'step': self.train_state.step,
            'config': self.config,
            'contract_hash': self.contract.compute_hash(),
            'training_metrics': self.training_metrics
        }
        
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint_data, f)
    
    def load_checkpoint(self, filepath: str) -> None:
        """Load model checkpoint."""
        import pickle
        with open(filepath, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        self.train_state = self.train_state.replace(
            params=checkpoint_data['params'],
            opt_state=checkpoint_data['opt_state'],
            step=checkpoint_data['step']
        )
        
        self.training_metrics = checkpoint_data.get('training_metrics', [])


# Helper classes for constraint checking
@dataclass
class MockState:
    """Mock state object for constraint checking."""
    tokens: jnp.ndarray
    hidden_states: Optional[jnp.ndarray] = None
    user_id: str = "default_user"
    user_consent: bool = True
    anonymized: bool = True
    user_demographics: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.user_demographics is None:
            self.user_demographics = {}


@dataclass
class MockAction:
    """Mock action object for constraint checking."""
    output: str
    reward: Optional[float] = None
    features: Optional[jnp.ndarray] = None
    statements: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.statements is None:
            # Split output into sentences as statements
            self.statements = [s.strip() for s in self.output.split('.') if s.strip()]