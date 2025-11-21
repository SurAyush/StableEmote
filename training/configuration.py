from dataclasses import dataclass


@dataclass
class TrainingConfig:
    pretrained_model_name_or_path: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    dataset_name: str = "valhalla/emoji-dataset"
    output_dir: str = "/content/sddata/finetune_lora"
    mixed_precision: str = "bf16"
    resume_from_checkpoint: str = None
    dataloader_num_workers: int = 8
    resolution: int = 256
    train_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    max_train_steps: int = 15000
    learning_rate: int = 1e-04
    max_grad_norm: int = 1
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 0
    push_to_hub: bool = True
    hub_model_id: str = "StableEmote-lora"
    report_to: str = "wanddb"
    logging_dir: str = "logs"
    checkpointing_steps: int = 500
    validation_prompt: str = "emoji of robot face"
    num_validation_images: int = 4
    validation_epochs: int = 1
    image_interpolation_mode: str = "lanczos"
    center_crop: bool = False
    random_flip: bool = False
    learning_rate: float = 1e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-8
    rank: int = 4
    xformers_memory_efficient_attention: bool = True
    gradient_checkpointing: bool = False
    seed: int = 42
