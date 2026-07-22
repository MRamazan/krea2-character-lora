import copy
import json
from pathlib import Path
import yaml

asset_manifest = json.loads((PATHS["config"] / "training_asset_manifest.json").read_text(encoding="utf-8"))
fingerprint = json.loads((PATHS["dataset_audit"] / "dataset_fingerprint.json").read_text(encoding="utf-8"))

training_configuration_path = PATHS["config"] / "train_krea2_lora.yaml"
smoke_configuration_path = PATHS["config"] / "train_krea2_lora_smoke.yaml"

sample_section = {
    "sampler": "flowmatch",
    "sample_every": USER_CONFIG["training_sample_every"],
    "sample_start_step": 0,
    "width": USER_CONFIG["inference_width"],
    "height": USER_CONFIG["inference_height"],
    "prompts": USER_CONFIG["evaluation_prompts"],
    "neg": USER_CONFIG["negative_prompt"],
    "seed": USER_CONFIG["evaluation_seeds"][0],
    "walk_seed": False,
    "guidance_scale": USER_CONFIG["raw_sample_guidance"],
    "sample_steps": USER_CONFIG["raw_sample_steps"],
    "network_multiplier": USER_CONFIG["primary_adapter_scale"],
}

configuration = {
    "job": "extension",
    "config": {
        "name": USER_CONFIG["run_name"],
        "process": [
            {
                "type": "sd_trainer",
                "training_folder": str(PATHS["checkpoints"]),
                "device": "cuda:0",
                "trigger_word": USER_CONFIG["trigger_word"],
                "network": {
                    "type": "lora",
                    "linear": USER_CONFIG["lora_rank"],
                    "linear_alpha": USER_CONFIG["lora_alpha"],
                    "transformer_only": True,
                    "all_layers": False,
                    "layer_offloading": USER_CONFIG["layer_offloading"],
                },
                "save": {
                    "dtype": "float16",
                    "save_every": USER_CONFIG["save_every"],
                    "max_step_saves_to_keep": USER_CONFIG["max_step_saves_to_keep"],
                    "save_format": "safetensors",
                    "push_to_hub": False,
                },
                "datasets": [
                    {
                        "folder_path": str(PATHS["dataset_training"]),
                        "caption_ext": "txt",
                        "resolution": USER_CONFIG["training_resolutions"],
                        "buckets": True,
                        "bucket_tolerance": 16,
                        "num_repeats": USER_CONFIG["dataset_repeats"],
                        "caption_dropout_rate": USER_CONFIG["caption_dropout_rate"],
                        "token_dropout_rate": USER_CONFIG["token_dropout_rate"],
                        "shuffle_tokens": USER_CONFIG["shuffle_tokens"],
                        "keep_tokens": USER_CONFIG["keep_tokens"],
                        "random_crop": False,
                        "random_scale": False,
                        "flip_x": USER_CONFIG["flip_x"],
                        "flip_y": False,
                        "cache_latents": False,
                        "cache_latents_to_disk": USER_CONFIG["cache_latents_to_disk"],
                        "cache_text_embeddings": USER_CONFIG["cache_text_embeddings"],
                        "num_workers": 2,
                        "prefetch_factor": 2,
                    }
                ],
                "train": {
                    "batch_size": USER_CONFIG["batch_size"],
                    "steps": USER_CONFIG["training_steps"],
                    "gradient_accumulation": USER_CONFIG["gradient_accumulation"],
                    "train_unet": True,
                    "train_text_encoder": False,
                    "train_refiner": False,
                    "train_turbo": False,
                    "gradient_checkpointing": True,
                    "noise_scheduler": "flowmatch",
                    "timestep_type": "sigmoid",
                    "optimizer": USER_CONFIG["optimizer"],
                    "optimizer_params": {"weight_decay": USER_CONFIG["weight_decay"]},
                    "lr": USER_CONFIG["learning_rate"],
                    "lr_scheduler": USER_CONFIG["lr_scheduler"],
                    "lr_scheduler_params": {},
                    "max_grad_norm": USER_CONFIG["max_grad_norm"],
                    "loss_target": "noise",
                    "loss_type": "mse",
                    "content_or_style": "balanced",
                    "prompt_dropout_prob": 0.0,
                    "cache_text_embeddings": USER_CONFIG["cache_text_embeddings"],
                    "skip_first_sample": True,
                    "disable_sampling": USER_CONFIG["disable_training_samples"],
                    "merge_network_on_save": False,
                    "dtype": USER_CONFIG["training_dtype"],
                },
                "model": {
                    "name_or_path": str(Path(asset_manifest["training_model"]["checkpoint_path"]).parent),
                    "arch": "krea2",
                    "dtype": USER_CONFIG["training_dtype"],
                    "vae_dtype": USER_CONFIG["training_dtype"],
                    "te_dtype": USER_CONFIG["training_dtype"],
                    "quantize": USER_CONFIG["quantize_transformer"],
                    "quantize_te": USER_CONFIG["quantize_text_encoder"],
                    "low_vram": USER_CONFIG["low_vram"],
                    "layer_offloading": USER_CONFIG["layer_offloading"],
                    "split_model_over_gpus": False,
                    "compile": False,
                    "model_kwargs": {
                        "checkpoint_filename": asset_manifest["training_model"]["checkpoint_filename"],
                        "text_encoder_path": asset_manifest["text_encoder"]["local_directory"],
                        "vae_path": asset_manifest["vae"]["local_directory"],
                        "max_text_length": USER_CONFIG["max_text_length"],
                    },
                },
                "sample": sample_section,
            }
        ],
    },
    "meta": {
        "name": "[name]",
        "version": "1.0",
        "project_name": USER_CONFIG["project_name"],
        "concept_type": USER_CONFIG["concept_type"],
        "trigger_word": USER_CONFIG["trigger_word"],
        "base_model": USER_CONFIG["training_model_repository"],
        "base_model_revision": asset_manifest["training_model"]["revision"],
        "dataset_fingerprint_sha256": fingerprint["dataset_fingerprint_sha256"],
    },
}

smoke_configuration = copy.deepcopy(configuration)
smoke_configuration["config"]["name"] = f"{USER_CONFIG['run_name']}_smoke"
smoke_process = smoke_configuration["config"]["process"][0]
smoke_process["training_folder"] = str(PATHS["smoke_checkpoints"])
smoke_process["save"]["save_every"] = 1
smoke_process["save"]["max_step_saves_to_keep"] = max(3, USER_CONFIG["smoke_test_steps"] + 1)
smoke_process["train"]["steps"] = USER_CONFIG["smoke_test_steps"]
smoke_process["train"]["disable_sampling"] = True
smoke_process["sample"]["sample_every"] = 1000000000

training_configuration_path.write_text(yaml.safe_dump(configuration, sort_keys=False, allow_unicode=True), encoding="utf-8")
smoke_configuration_path.write_text(yaml.safe_dump(smoke_configuration, sort_keys=False, allow_unicode=True), encoding="utf-8")

print(f"Production configuration: {training_configuration_path}")
print(f"Smoke configuration: {smoke_configuration_path}")
print(training_configuration_path.read_text(encoding="utf-8"))