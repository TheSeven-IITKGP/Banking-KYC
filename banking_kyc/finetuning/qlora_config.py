"""QLoRA and LoRA Experiment Configurations and Ablation Matrix.

Author: chandan bauri <chandanbauri441@gmail.com>
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class QLoRAConfig:
    """Hyperparameter specification for a single fine-tuning run."""

    experiment_name: str
    base_model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    quantization_type: str = "nf4"  # 4-bit NormalFloat
    use_double_quant: bool = True
    compute_dtype: str = "bfloat16"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    max_seq_length: int = 1024
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 3
    weight_decay: float = 0.01
    optim: str = "paged_adamw_8bit"
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class QLoRAExperimentConfigs:
    """Production matrix containing 2 configurations and 2 ablation variations."""

    @classmethod
    def get_config_1_balanced(cls) -> QLoRAConfig:
        """Config 1: Production balanced QLoRA (Rank 16, Alpha 32)."""
        return QLoRAConfig(
            experiment_name="config_1_balanced_qlora_r16",
            base_model="meta-llama/Meta-Llama-3-8B-Instruct",
            lora_r=16,
            lora_alpha=32,
            learning_rate=2e-4,
            description="Optimal balance between model capacity, parameter efficiency, and generalization on noisy OCR text.",
        )

    @classmethod
    def get_config_2_expressive(cls) -> QLoRAConfig:
        """Config 2: High-Rank Expressive QLoRA (Rank 32, Alpha 64)."""
        return QLoRAConfig(
            experiment_name="config_2_expressive_qlora_r32",
            base_model="meta-llama/Meta-Llama-3-8B-Instruct",
            lora_r=32,
            lora_alpha=64,
            learning_rate=1.5e-4,
            lora_dropout=0.10,
            description="Higher parameter budget to evaluate whether complex spatial OCR address layouts benefit from increased rank.",
        )

    @classmethod
    def get_ablation_1_low_rank(cls) -> QLoRAConfig:
        """Ablation 1: Low-Rank Bottleneck (Rank 8, Alpha 16)."""
        return QLoRAConfig(
            experiment_name="ablation_1_low_rank_r8",
            base_model="meta-llama/Meta-Llama-3-8B-Instruct",
            lora_r=8,
            lora_alpha=16,
            learning_rate=3e-4,
            description="Ablation testing if low-rank adaptation suffers under heavy OCR noise or multi-lingual address lines.",
        )

    @classmethod
    def get_ablation_2_no_ocr_noise(cls) -> QLoRAConfig:
        """Ablation 2: Clean Synthetic Data Only (No OCR Noise Augmentation)."""
        return QLoRAConfig(
            experiment_name="ablation_2_clean_data_only",
            base_model="meta-llama/Meta-Llama-3-8B-Instruct",
            lora_r=16,
            lora_alpha=32,
            learning_rate=2e-4,
            description="Ablation proving that training on simulated OCR character substitutions and noise is critical for field robustness.",
        )

    @classmethod
    def list_all_configs(cls) -> Dict[str, QLoRAConfig]:
        return {
            "config_1_balanced": cls.get_config_1_balanced(),
            "config_2_expressive": cls.get_config_2_expressive(),
            "ablation_1_low_rank": cls.get_ablation_1_low_rank(),
            "ablation_2_no_ocr_noise": cls.get_ablation_2_no_ocr_noise(),
        }
