# W&B logging setup
import wandb
import torch
import numpy as np


def log_validation(
    logger, pipeline, config, accelerator, epoch, is_final_validation=False
):
    logger.info(
        f"Running validation... \n Generating {config.num_validation_images} images with prompt:"
        f" {config.validation_prompt}."
    )

    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(
        disable=True
    )  # freezes progress bar for validation
    generator = torch.Generator(device=accelerator.device)
    if config.seed is not None:
        generator = generator.manual_seed(config.seed)

    images = []
    autocast_ctx = torch.autocast(accelerator.device.type)  # automatic mixed precision

    with autocast_ctx:
        for _ in range(config.num_validation_images):
            images.append(
                pipeline(
                    config.validation_prompt,
                    num_inference_steps=40,
                    generator=generator,
                ).images[0]
            )

    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(phase_name, np_images, epoch, dataformats="NHWC")

        if tracker.name == "wandb":
            tracker.log(
                {
                    phase_name: [
                        wandb.Image(image, caption=f"{i}: {config.validation_prompt}")
                        for i, image in enumerate(images)
                    ]
                }
            )

    return images


def log_init(logger, len_train_dataset, config, total_batch_size):
    logger.info("***** Training *****")
    logger.info(f"  Num examples = {len_train_dataset}")
    logger.info(f"  Num Epochs = {config.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {config.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {config.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {config.max_train_steps}")
