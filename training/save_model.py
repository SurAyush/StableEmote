import os
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card


def save_model_card(repo_id: str, base_model: str = None, repo_folder: str = None):
    model_description = """
        # LoRA text2image fine-tuning of stable-diffusion-1.5
        These are LoRA adaption weights for stable-diffusion-1.5. The weights were fine-tuned on the valhalla/emoji-dataset dataset.
        Refer to the GitHub Repo for more information: https://github.com/SurAyush/StableEmote
    """

    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        base_model=base_model,
        model_description=model_description,
        inference=True,
    )

    tags = ["text-to-image", "diffusers", "stable-diffusion", "lora"]

    model_card = populate_model_card(model_card, tags=tags)

    # Save the model card to the repo folder
    model_card.save(os.path.join(repo_folder, "README.md"))
