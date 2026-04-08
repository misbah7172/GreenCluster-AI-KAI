"""
Active Inference Module (Non-DRL)
Learns from environment and machine state to adjust decisions in real-time.

Key Features:
- Bayesian belief updating (not neural networks)
- Expected Free Energy minimization
- Active sampling for uncertainty reduction
- Real-time decision adaptation
"""

import math
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import random


@dataclass
class BeliefState:
    """Probabilistic belief about system state."""
    # Belief distribution over discrete states
    state_probs: Dict[str, float] = field(default_factory=dict)
    
    # Continuous parameters with uncertainty
    mean_values: Dict[str, float] = field(default_factory=dict)
    variances: Dict[str, float] = field(default_factory=dict)
    
    # Confidence level (0-1)
    confidence: float = 0.5
    last_update: float = field(default_factory=time.time)
    
    def get_entropy(self) -> float:
        """Calculate entropy of belief distribution."""
        if not self.state_probs:
            return 0.0
        entropy = 0.0
        for p in self.state_probs.values():
            if p > 0:
                entropy -= p * math.log(p + 1e-10)
        return entropy
    
    def normalize(self) -> None:
        """Normalize probability distribution."""
        total = sum(self.state_probs.values())
        if total > 0:
            for key in self.state_probs:
                self.state_probs[key] /= total


@dataclass
class Observation:
    """Observation from environment."""
    timestamp: float
    observation_type: str  # e.g., "latency", "power", "throughput"
    value: float
    source: str  # e.g., "node_1", "gpu_0"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Action:
    """Action to take on the system."""
    action_type: str  # e.g., "adjust_batch", "migrate_layer", "change_precision"
    parameters: Dict[str, Any] = field(default_factory=dict)
    expected_outcome: Optional[str] = None
    confidence: float = 0.5


@dataclass
class PolicyPreference:
    """Preferences that guide policy selection."""
    target_latency_ms: float = 100.0
    target_throughput: float = 10.0  # tokens/sec
    max_power_watts: float = 1000.0
    min_accuracy: float = 0.95
    
    # Weights for multi-objective optimization
    latency_weight: float = 0.3
    throughput_weight: float = 0.3
    power_weight: float = 0.2
    accuracy_weight: float = 0.2


class GenerativeModel:
    """
    Generative model for active inference.
    Models P(observations | states) and P(states | actions).
    """
    
    def __init__(self):
        # Likelihood model: P(observation | state)
        # Maps (state, obs_type) -> (mean, variance)
        self.likelihood: Dict[Tuple[str, str], Tuple[float, float]] = {}
        
        # Transition model: P(next_state | state, action)
        # Maps (state, action) -> {next_state: prob}
        self.transitions: Dict[Tuple[str, str], Dict[str, float]] = {}
        
        # Prior preferences over observations
        self.preferred_observations: Dict[str, float] = {}
        
        # Learning rates
        self.likelihood_lr = 0.1
        self.transition_lr = 0.1
    
    def update_likelihood(
        self, 
        state: str, 
        obs_type: str, 
        observed_value: float
    ) -> None:
        """Update likelihood model with new observation."""
        key = (state, obs_type)
        
        if key not in self.likelihood:
            self.likelihood[key] = (observed_value, 1.0)
            return
        
        old_mean, old_var = self.likelihood[key]
        
        # Bayesian update (simplified Kalman-like)
        new_mean = old_mean + self.likelihood_lr * (observed_value - old_mean)
        new_var = old_var + self.likelihood_lr * ((observed_value - new_mean)**2 - old_var)
        new_var = max(0.01, new_var)  # Minimum variance
        
        self.likelihood[key] = (new_mean, new_var)
    
    def update_transition(
        self,
        state: str,
        action: str,
        next_state: str
    ) -> None:
        """Update transition model with observed state change."""
        key = (state, action)
        
        if key not in self.transitions:
            self.transitions[key] = defaultdict(float)
        
        # Increment count for observed transition
        for s in self.transitions[key]:
            self.transitions[key][s] *= (1 - self.transition_lr)
        
        self.transitions[key][next_state] += self.transition_lr
        
        # Normalize
        total = sum(self.transitions[key].values())
        if total > 0:
            for s in self.transitions[key]:
                self.transitions[key][s] /= total
    
    def predict_observation(
        self, 
        state: str, 
        obs_type: str
    ) -> Tuple[float, float]:
        """Predict observation given state."""
        key = (state, obs_type)
        if key in self.likelihood:
            return self.likelihood[key]
        return (0.0, 1.0)  # Uninformative prior
    
    def predict_next_state(
        self, 
        state: str, 
        action: str
    ) -> Dict[str, float]:
        """Predict next state distribution given action."""
        key = (state, action)
        if key in self.transitions:
            return dict(self.transitions[key])
        return {state: 1.0}  # Default: stay in same state


class ActiveInferenceAgent:
    """
    Active Inference agent for real-time system optimization.
    
    Uses Expected Free Energy (EFE) minimization to:
    1. Reduce uncertainty about system state (epistemic value)
    2. Achieve preferred outcomes (pragmatic value)
    
    This is NOT deep reinforcement learning - it uses:
    - Bayesian belief updating
    - Variational inference
    - Free energy principle
    """
    
    def __init__(
        self,
        preferences: Optional[PolicyPreference] = None,
        planning_horizon: int = 3,
        num_action_samples: int = 10,
        exploration_factor: float = 0.2,
    ):
        """
        Initialize active inference agent.
        
        Args:
            preferences: Target preferences for optimization
            planning_horizon: How many steps to look ahead
            num_action_samples: Actions to evaluate per step
            exploration_factor: Weight for epistemic (exploratory) value
        """
        self.preferences = preferences or PolicyPreference()
        self.planning_horizon = planning_horizon
        self.num_action_samples = num_action_samples
        self.exploration_factor = exploration_factor
        
        # Internal models
        self.generative_model = GenerativeModel()
        self.belief = BeliefState()
        
        # State space (discrete states for simplicity)
        self.states = [
            "optimal", "high_latency", "high_power", 
            "low_throughput", "degraded", "critical"
        ]
        
        # Action space
        self.actions = [
            "increase_batch", "decrease_batch",
            "increase_precision", "decrease_precision",
            "enable_offloading", "disable_offloading",
            "migrate_layer", "no_action"
        ]
        
        # Observation types we track
        self.obs_types = ["latency", "power", "throughput", "accuracy", "memory"]
        
        # History for learning
        self.observation_history: List[Observation] = []
        self.action_history: List[Tuple[Action, str]] = []
        
        # Initialize uniform beliefs
        for state in self.states:
            self.belief.state_probs[state] = 1.0 / len(self.states)
        
        self._lock = threading.Lock()
    
    def observe(self, observation: Observation) -> None:
        """
        Process new observation and update beliefs.
        
        Uses Bayesian belief updating:
        P(state | obs) ∝ P(obs | state) * P(state)
        """
        with self._lock:
            self.observation_history.append(observation)
            
            # Keep history bounded
            if len(self.observation_history) > 1000:
                self.observation_history = self.observation_history[-1000:]
            
            # Update belief using likelihood
            self._update_belief(observation)
            
            # Update generative model
            current_state = self._get_most_likely_state()
            self.generative_model.update_likelihood(
                current_state, 
                observation.observation_type, 
                observation.value
            )
    
    def _update_belief(self, observation: Observation) -> None:
        """Bayesian belief update given observation."""
        obs_type = observation.observation_type
        obs_value = observation.value
        
        new_probs = {}
        total = 0.0
        
        for state in self.states:
            # Get likelihood P(obs | state)
            mean, var = self.generative_model.predict_observation(state, obs_type)
            
            # Gaussian likelihood
            likelihood = math.exp(-0.5 * ((obs_value - mean) ** 2) / (var + 1e-6))
            likelihood /= math.sqrt(2 * math.pi * var + 1e-6)
            
            # Prior
            prior = self.belief.state_probs.get(state, 1.0 / len(self.states))
            
            # Posterior (unnormalized)
            posterior = likelihood * prior
            new_probs[state] = posterior
            total += posterior
        
        # Normalize
        if total > 0:
            for state in new_probs:
                new_probs[state] /= total
        
        self.belief.state_probs = new_probs
        self.belief.last_update = time.time()
        
        # Update confidence based on entropy
        entropy = self.belief.get_entropy()
        max_entropy = math.log(len(self.states))
        self.belief.confidence = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.5
    
    def _get_most_likely_state(self) -> str:
        """Get state with highest probability."""
        if not self.belief.state_probs:
            return "optimal"
        return max(self.belief.state_probs.items(), key=lambda x: x[1])[0]
    
    def select_action(self) -> Action:
        """
        Select action using Expected Free Energy minimization.
        
        EFE = Expected Pragmatic Value + Expected Epistemic Value
        
        Pragmatic: How well does action achieve preferences?
        Epistemic: How much does action reduce uncertainty?
        """
        with self._lock:
            current_state = self._get_most_likely_state()
            
            best_action = None
            best_efe = float('inf')
            
            for action_type in self.actions:
                efe = self._compute_expected_free_energy(
                    current_state, 
                    action_type,
                    horizon=self.planning_horizon
                )
                
                if efe < best_efe:
                    best_efe = efe
                    best_action = action_type
            
            # Create action object
            action = Action(
                action_type=best_action or "no_action",
                parameters=self._get_action_parameters(best_action or "no_action"),
                expected_outcome=self._predict_outcome(current_state, best_action or "no_action"),
                confidence=self.belief.confidence,
            )
            
            return action
    
    def _compute_expected_free_energy(
        self, 
        state: str, 
        action: str,
        horizon: int = 1
    ) -> float:
        """
        Compute Expected Free Energy for an action.
        
        G = D_KL[Q(o|π) || P(o)] - E_Q[H[P(s|o)]]
        
        First term: pragmatic value (preference satisfaction)
        Second term: epistemic value (information gain)
        """
        # Predict next state distribution
        next_state_dist = self.generative_model.predict_next_state(state, action)
        
        pragmatic_value = 0.0
        epistemic_value = 0.0
        
        for next_state, prob in next_state_dist.items():
            if prob < 1e-6:
                continue
            
            # Pragmatic: KL divergence from preferred observations
            for obs_type in self.obs_types:
                predicted_mean, predicted_var = self.generative_model.predict_observation(
                    next_state, obs_type
                )
                preferred = self._get_preferred_value(obs_type)
                
                # KL-like term: how far from preference?
                deviation = (predicted_mean - preferred) ** 2 / (predicted_var + 1e-6)
                pragmatic_value += prob * deviation * self._get_obs_weight(obs_type)
            
            # Epistemic: expected information gain
            # Higher variance = more uncertainty = more epistemic value
            for obs_type in self.obs_types:
                _, var = self.generative_model.predict_observation(next_state, obs_type)
                epistemic_value += prob * math.log(var + 1e-6)
        
        # Combine with exploration factor
        efe = pragmatic_value - self.exploration_factor * epistemic_value
        
        # Recursive planning for horizon > 1
        if horizon > 1:
            for next_state, prob in next_state_dist.items():
                if prob < 1e-6:
                    continue
                # Find best action from next state
                min_future_efe = min(
                    self._compute_expected_free_energy(next_state, a, horizon - 1)
                    for a in self.actions
                )
                efe += prob * 0.9 * min_future_efe  # Discount factor
        
        return efe
    
    def _get_preferred_value(self, obs_type: str) -> float:
        """Get preferred observation value."""
        prefs = {
            "latency": self.preferences.target_latency_ms,
            "throughput": self.preferences.target_throughput,
            "power": self.preferences.max_power_watts * 0.7,  # Prefer lower power
            "accuracy": self.preferences.min_accuracy,
            "memory": 0.7,  # 70% utilization
        }
        return prefs.get(obs_type, 0.0)
    
    def _get_obs_weight(self, obs_type: str) -> float:
        """Get weight for observation type."""
        weights = {
            "latency": self.preferences.latency_weight,
            "throughput": self.preferences.throughput_weight,
            "power": self.preferences.power_weight,
            "accuracy": self.preferences.accuracy_weight,
            "memory": 0.1,
        }
        return weights.get(obs_type, 0.1)
    
    def _get_action_parameters(self, action_type: str) -> Dict[str, Any]:
        """Get default parameters for action type."""
        params = {
            "increase_batch": {"delta": 2},
            "decrease_batch": {"delta": -2},
            "increase_precision": {"target": "fp16"},
            "decrease_precision": {"target": "int8"},
            "enable_offloading": {"threshold": 0.8},
            "disable_offloading": {},
            "migrate_layer": {"count": 1},
            "no_action": {},
        }
        return params.get(action_type, {})
    
    def _predict_outcome(self, state: str, action: str) -> str:
        """Predict most likely outcome state."""
        next_dist = self.generative_model.predict_next_state(state, action)
        if next_dist:
            return max(next_dist.items(), key=lambda x: x[1])[0]
        return state
    
    def record_action_outcome(self, action: Action, outcome_state: str) -> None:
        """Record action outcome for learning."""
        with self._lock:
            current_state = self._get_most_likely_state()
            
            # Update transition model
            self.generative_model.update_transition(
                current_state,
                action.action_type,
                outcome_state
            )
            
            self.action_history.append((action, outcome_state))
            
            # Keep bounded
            if len(self.action_history) > 500:
                self.action_history = self.action_history[-500:]
    
    def get_uncertainty_metrics(self) -> Dict[str, float]:
        """Get current uncertainty metrics."""
        with self._lock:
            return {
                "belief_entropy": self.belief.get_entropy(),
                "confidence": self.belief.confidence,
                "most_likely_state": self._get_most_likely_state(),
                "state_probability": self.belief.state_probs.get(
                    self._get_most_likely_state(), 0
                ),
                "observation_count": len(self.observation_history),
                "action_count": len(self.action_history),
            }
    
    def get_state_beliefs(self) -> Dict[str, float]:
        """Get current belief distribution over states."""
        with self._lock:
            return dict(self.belief.state_probs)
    
    def reset_beliefs(self) -> None:
        """Reset beliefs to uniform distribution."""
        with self._lock:
            for state in self.states:
                self.belief.state_probs[state] = 1.0 / len(self.states)
            self.belief.confidence = 0.5


class ActiveInferenceController:
    """
    Controller that uses Active Inference for system optimization.
    Integrates with KAI monitoring and execution systems.
    """
    
    def __init__(
        self,
        agent: Optional[ActiveInferenceAgent] = None,
        action_interval_s: float = 5.0,
    ):
        self.agent = agent or ActiveInferenceAgent()
        self.action_interval_s = action_interval_s
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._action_callback: Optional[callable] = None
    
    def start(self, action_callback: callable) -> None:
        """Start the active inference control loop."""
        self._action_callback = action_callback
        self._running = True
        self._thread = threading.Thread(target=self._control_loop, daemon=True)
        self._thread.start()
    
    def stop(self) -> None:
        """Stop the control loop."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
    
    def _control_loop(self) -> None:
        """Main control loop."""
        while self._running:
            try:
                action = self.agent.select_action()
                
                if action.action_type != "no_action" and self._action_callback:
                    self._action_callback(action)
                
                time.sleep(self.action_interval_s)
            except Exception as e:
                print(f"Active inference error: {e}")
                time.sleep(1.0)
    
    def feed_observation(
        self,
        obs_type: str,
        value: float,
        source: str = "system"
    ) -> None:
        """Feed observation to the agent."""
        obs = Observation(
            timestamp=time.time(),
            observation_type=obs_type,
            value=value,
            source=source,
        )
        self.agent.observe(obs)


# Plugin registration
try:
    from model.plugin_architecture import PluginRegistry
    
    @PluginRegistry.register("optimizer", "active_inference")
    class ActiveInferencePlugin:
        """Active Inference Optimizer Plugin."""
        
        def __init__(self, **kwargs):
            self.agent = ActiveInferenceAgent(**kwargs)
            self.controller = ActiveInferenceController(self.agent)
        
        def observe(self, obs_type: str, value: float, source: str = "system"):
            obs = Observation(
                timestamp=time.time(),
                observation_type=obs_type,
                value=value,
                source=source,
            )
            self.agent.observe(obs)
        
        def get_action(self) -> Action:
            return self.agent.select_action()

except ImportError:
    pass
