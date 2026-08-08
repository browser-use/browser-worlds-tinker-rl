"""Training configuration built on Tinker Cookbook's shared RL engine."""

from __future__ import annotations

from dataclasses import dataclass

from tinker_cookbook.rl.train import Config


@dataclass(frozen=True)
class BrowserTrainingConfig:
    model_name: str
    group_size: int = 4
    groups_per_batch: int = 4
    learning_rate: float = 1e-5
    lora_rank: int = 32
    max_tokens: int = 8192
    temperature: float = 1.0
    kl_penalty_coef: float = 0.0
    log_path: str = "/tmp/browser-worlds-tinker-rl"


def build_tinker_config(config: BrowserTrainingConfig, dataset_builder: object) -> Config:
    """Create the same Config used by Harbor RL with a browser dataset builder."""
    return Config(
        learning_rate=config.learning_rate,
        dataset_builder=dataset_builder,
        model_name=config.model_name,
        recipe_name="browser_worlds_tinker_rl",
        lora_rank=config.lora_rank,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        kl_penalty_coef=config.kl_penalty_coef,
        log_path=config.log_path,
    )
