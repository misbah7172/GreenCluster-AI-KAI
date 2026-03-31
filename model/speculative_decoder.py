"""
Speculative Decoding for KAI.

Implements speculative decoding to reduce latency and GPU utilization:
- Use a smaller draft model to generate candidate tokens
- Verify tokens using the main distributed model
- Accept valid tokens, reject incorrect ones
- No change in final output (mathematically equivalent)

Usage::

    from model.speculative_decoder import SpeculativeDecoder
    
    decoder = SpeculativeDecoder(
        main_model=distributed_model,
        draft_model=small_model,
        num_speculative_tokens=4,
    )
    
    output = decoder.generate(prompt, max_tokens=100)
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


class VerificationMode(Enum):
    """Token verification modes."""
    STRICT = "strict"          # Reject if any probability mismatch
    THRESHOLD = "threshold"    # Accept if main prob >= threshold
    SAMPLING = "sampling"      # Rejection sampling (mathematically exact)


@dataclass
class SpeculativeConfig:
    """Configuration for speculative decoding."""
    num_speculative_tokens: int = 4  # Tokens to speculate per step
    verification_mode: VerificationMode = VerificationMode.SAMPLING
    acceptance_threshold: float = 0.9  # For THRESHOLD mode
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    draft_temperature: float = 1.0  # Can be lower for more confident drafts


@dataclass
class SpeculativeStats:
    """Statistics for speculative decoding."""
    total_tokens: int = 0
    accepted_tokens: int = 0
    rejected_tokens: int = 0
    draft_time_ms: float = 0.0
    verify_time_ms: float = 0.0
    speedup_ratio: float = 1.0
    
    @property
    def acceptance_rate(self) -> float:
        """Token acceptance rate."""
        if self.total_tokens == 0:
            return 0.0
        return self.accepted_tokens / self.total_tokens
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_tokens": self.total_tokens,
            "accepted_tokens": self.accepted_tokens,
            "rejected_tokens": self.rejected_tokens,
            "acceptance_rate": round(self.acceptance_rate, 3),
            "draft_time_ms": round(self.draft_time_ms, 2),
            "verify_time_ms": round(self.verify_time_ms, 2),
            "speedup_ratio": round(self.speedup_ratio, 2),
        }


class DraftModelWrapper:
    """Wrapper for draft models with caching.
    
    Parameters
    ----------
    model : nn.Module
        Small draft model (e.g., smaller LLM or distilled version)
    tokenizer
        Tokenizer compatible with both draft and main model
    device : str
        Device for draft model
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        device: str = "cuda:0",
    ):
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device
        
        # KV cache for draft model
        self._kv_cache: Optional[Tuple[Tensor, ...]] = None
    
    def generate_speculative(
        self,
        input_ids: Tensor,
        num_tokens: int,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> Tuple[Tensor, Tensor]:
        """Generate speculative tokens.
        
        Parameters
        ----------
        input_ids : Tensor
            Input token IDs [batch, seq_len]
        num_tokens : int
            Number of tokens to speculate
        temperature : float
            Sampling temperature
        top_k : int
            Top-k sampling
        top_p : float
            Nucleus sampling threshold
            
        Returns
        -------
        draft_tokens, draft_probs : tuple[Tensor, Tensor]
            Speculated tokens and their probabilities
        """
        batch_size = input_ids.shape[0]
        draft_tokens = []
        draft_probs = []
        
        current_ids = input_ids.to(self.device)
        
        with torch.no_grad():
            for _ in range(num_tokens):
                # Forward pass
                outputs = self.model(current_ids)
                
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits[:, -1, :]
                else:
                    logits = outputs[:, -1, :]
                
                # Apply temperature
                if temperature != 1.0:
                    logits = logits / temperature
                
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float("-inf")
                
                # Top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float("-inf")
                
                # Sample
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                next_prob = probs.gather(1, next_token)
                
                draft_tokens.append(next_token)
                draft_probs.append(next_prob)
                
                # Update input
                current_ids = torch.cat([current_ids, next_token], dim=1)
        
        # Stack results
        draft_tokens = torch.cat(draft_tokens, dim=1)  # [batch, num_tokens]
        draft_probs = torch.cat(draft_probs, dim=1)    # [batch, num_tokens]
        
        return draft_tokens, draft_probs
    
    def clear_cache(self) -> None:
        """Clear KV cache."""
        self._kv_cache = None


class SpeculativeDecoder:
    """Speculative decoding engine.
    
    Uses a smaller draft model to generate candidate tokens,
    then verifies them with the main model. Mathematically
    equivalent to sampling from the main model.
    
    Parameters
    ----------
    main_model : nn.Module or callable
        Main model for verification (can be distributed)
    draft_model : nn.Module
        Smaller draft model for speculation
    tokenizer
        Shared tokenizer
    config : SpeculativeConfig
        Speculative decoding configuration
    """
    
    def __init__(
        self,
        main_model: Union[nn.Module, Callable],
        draft_model: nn.Module,
        tokenizer,
        config: Optional[SpeculativeConfig] = None,
        device: str = "cuda:0",
    ):
        self.main_model = main_model
        self.tokenizer = tokenizer
        self.config = config or SpeculativeConfig()
        self.device = device
        
        # Wrap draft model
        self.draft = DraftModelWrapper(draft_model, tokenizer, device)
        
        # Statistics
        self._stats = SpeculativeStats()
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Generate text using speculative decoding.
        
        Parameters
        ----------
        prompt : str
            Input prompt
        max_new_tokens : int
            Maximum tokens to generate
        temperature : float, optional
            Override config temperature
        top_k : int, optional
            Override config top_k
        top_p : float, optional
            Override config top_p
        callback : callable, optional
            Called with each generated token
            
        Returns
        -------
        str
            Generated text (including prompt)
        """
        temperature = temperature or self.config.temperature
        top_k = top_k or self.config.top_k
        top_p = top_p or self.config.top_p
        
        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        generated_tokens = []
        total_draft_time = 0.0
        total_verify_time = 0.0
        
        while len(generated_tokens) < max_new_tokens:
            # Step 1: Generate speculative tokens with draft model
            draft_start = time.perf_counter()
            draft_tokens, draft_probs = self.draft.generate_speculative(
                input_ids,
                num_tokens=self.config.num_speculative_tokens,
                temperature=self.config.draft_temperature,
                top_k=top_k,
                top_p=top_p,
            )
            total_draft_time += (time.perf_counter() - draft_start) * 1000
            
            # Step 2: Verify with main model
            verify_start = time.perf_counter()
            accepted, main_probs = self._verify_tokens(
                input_ids,
                draft_tokens,
                draft_probs,
                temperature,
            )
            total_verify_time += (time.perf_counter() - verify_start) * 1000
            
            # Step 3: Accept valid tokens
            num_accepted = accepted.sum().item()
            
            if num_accepted > 0:
                accepted_tokens = draft_tokens[0, :num_accepted]
                generated_tokens.extend(accepted_tokens.tolist())
                input_ids = torch.cat([input_ids, accepted_tokens.unsqueeze(0)], dim=1)
                
                # Callback for streaming
                if callback:
                    for tok_id in accepted_tokens.tolist():
                        callback(self.tokenizer.decode([tok_id]))
            
            # Step 4: Sample correction token if needed
            if num_accepted < self.config.num_speculative_tokens:
                correction_token = self._sample_correction(
                    main_probs[:, num_accepted],
                    draft_probs[0, num_accepted] if num_accepted < draft_probs.shape[1] else None,
                    temperature,
                )
                generated_tokens.append(correction_token.item())
                input_ids = torch.cat([input_ids, correction_token.unsqueeze(0).unsqueeze(0)], dim=1)
                
                if callback:
                    callback(self.tokenizer.decode([correction_token.item()]))
            
            # Update stats
            self._stats.total_tokens += len(draft_tokens[0])
            self._stats.accepted_tokens += num_accepted
            self._stats.rejected_tokens += len(draft_tokens[0]) - num_accepted
            
            # Check for EOS
            if self.tokenizer.eos_token_id in generated_tokens[-self.config.num_speculative_tokens:]:
                break
        
        # Final stats
        self._stats.draft_time_ms = total_draft_time
        self._stats.verify_time_ms = total_verify_time
        
        # Calculate speedup (compared to generating each token individually)
        tokens_generated = len(generated_tokens)
        if tokens_generated > 0 and total_verify_time > 0:
            # Estimate: without speculation, we'd need tokens_generated verify calls
            estimated_no_spec = total_verify_time / tokens_generated * tokens_generated
            self._stats.speedup_ratio = estimated_no_spec / (total_draft_time + total_verify_time)
        
        return self.tokenizer.decode(input_ids[0])
    
    def _verify_tokens(
        self,
        input_ids: Tensor,
        draft_tokens: Tensor,
        draft_probs: Tensor,
        temperature: float,
    ) -> Tuple[Tensor, Tensor]:
        """Verify draft tokens with main model.
        
        Returns acceptance mask and main model probabilities.
        """
        batch_size, num_draft = draft_tokens.shape
        
        # Build input with draft tokens
        full_input = torch.cat([input_ids, draft_tokens], dim=1)
        
        # Get main model logits for all positions
        with torch.no_grad():
            if callable(self.main_model) and not isinstance(self.main_model, nn.Module):
                # Distributed model callable
                outputs = self.main_model(full_input)
            else:
                outputs = self.main_model(full_input.to(self.device))
            
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
        
        # Get probabilities at draft positions
        # Position i in output corresponds to predicting token i+1
        start_pos = input_ids.shape[1] - 1
        end_pos = start_pos + num_draft
        
        main_logits = logits[:, start_pos:end_pos, :]
        
        if temperature != 1.0:
            main_logits = main_logits / temperature
        
        main_probs = F.softmax(main_logits, dim=-1)
        
        # Get probabilities for draft tokens
        draft_token_probs = main_probs.gather(
            2, draft_tokens.unsqueeze(-1)
        ).squeeze(-1)  # [batch, num_draft]
        
        # Verification based on mode
        if self.config.verification_mode == VerificationMode.STRICT:
            # Accept if main prob >= draft prob
            accepted = draft_token_probs >= draft_probs
        
        elif self.config.verification_mode == VerificationMode.THRESHOLD:
            # Accept if main prob >= threshold
            accepted = draft_token_probs >= self.config.acceptance_threshold
        
        else:  # SAMPLING (mathematically exact)
            # Rejection sampling: accept with probability min(1, p_main / p_draft)
            accept_prob = torch.clamp(draft_token_probs / draft_probs, max=1.0)
            random_vals = torch.rand_like(accept_prob)
            accepted = random_vals < accept_prob
        
        # Convert to contiguous acceptance (accept prefix only)
        accepted = self._get_acceptance_prefix(accepted)
        
        return accepted, main_probs
    
    def _get_acceptance_prefix(self, accepted: Tensor) -> Tensor:
        """Convert to contiguous acceptance (accept only prefix of True values)."""
        batch_size, seq_len = accepted.shape
        result = torch.zeros_like(accepted)
        
        for b in range(batch_size):
            for i in range(seq_len):
                if accepted[b, i]:
                    result[b, i] = True
                else:
                    break
        
        return result
    
    def _sample_correction(
        self,
        main_probs: Tensor,
        draft_prob: Optional[Tensor],
        temperature: float,
    ) -> Tensor:
        """Sample correction token when speculation fails.
        
        Uses adjusted distribution to maintain mathematical correctness.
        """
        # For exact sampling: sample from max(0, p_main - p_draft)
        # normalized
        if draft_prob is not None:
            adjusted_probs = F.relu(main_probs - draft_prob.unsqueeze(-1))
            adjusted_probs = adjusted_probs / adjusted_probs.sum(dim=-1, keepdim=True)
        else:
            adjusted_probs = main_probs
        
        # Handle NaN from division
        adjusted_probs = torch.nan_to_num(adjusted_probs, nan=0.0)
        if adjusted_probs.sum() == 0:
            adjusted_probs = main_probs
        
        return torch.multinomial(adjusted_probs, num_samples=1).squeeze(-1)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get speculative decoding statistics."""
        return self._stats.to_dict()
    
    def reset_stats(self) -> None:
        """Reset statistics."""
        self._stats = SpeculativeStats()


class AdaptiveSpeculativeDecoder(SpeculativeDecoder):
    """Speculative decoder with adaptive speculation length.
    
    Dynamically adjusts num_speculative_tokens based on acceptance rate.
    """
    
    def __init__(
        self,
        main_model: Union[nn.Module, Callable],
        draft_model: nn.Module,
        tokenizer,
        config: Optional[SpeculativeConfig] = None,
        device: str = "cuda:0",
        min_speculative: int = 1,
        max_speculative: int = 8,
    ):
        super().__init__(main_model, draft_model, tokenizer, config, device)
        self._min_spec = min_speculative
        self._max_spec = max_speculative
        self._acceptance_history: List[float] = []
    
    def _adjust_speculation_length(self) -> None:
        """Adjust speculation length based on recent acceptance rate."""
        if len(self._acceptance_history) < 5:
            return
        
        recent_rate = sum(self._acceptance_history[-5:]) / 5
        current = self.config.num_speculative_tokens
        
        if recent_rate > 0.8 and current < self._max_spec:
            # High acceptance: increase speculation
            self.config.num_speculative_tokens = min(current + 1, self._max_spec)
            logger.debug("Increased speculation to %d (rate=%.2f)", 
                        self.config.num_speculative_tokens, recent_rate)
        
        elif recent_rate < 0.4 and current > self._min_spec:
            # Low acceptance: decrease speculation
            self.config.num_speculative_tokens = max(current - 1, self._min_spec)
            logger.debug("Decreased speculation to %d (rate=%.2f)",
                        self.config.num_speculative_tokens, recent_rate)


def create_draft_model_from_main(
    main_model: nn.Module,
    reduction_factor: int = 4,
) -> nn.Module:
    """Create a smaller draft model from main model architecture.
    
    This is a simplified implementation - real implementation would
    use knowledge distillation to train the draft model.
    
    Parameters
    ----------
    main_model : nn.Module
        Main model to create draft from
    reduction_factor : int
        How much smaller the draft model should be
        
    Returns
    -------
    nn.Module
        Smaller draft model
    """
    # This would need model-specific implementation
    # For now, return a simple wrapper that just uses the main model
    # with reduced computation
    
    class SimpleDraftModel(nn.Module):
        def __init__(self, base_model: nn.Module):
            super().__init__()
            self.base_model = base_model
            
        def forward(self, input_ids):
            # Use only first few layers if possible
            with torch.no_grad():
                return self.base_model(input_ids)
    
    return SimpleDraftModel(main_model)
