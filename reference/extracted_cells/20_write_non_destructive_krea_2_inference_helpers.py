from pathlib import Path

helper_path = PATHS["helpers"] / "krea2_runtime.py"
helper_source = r"""
import gc
import hashlib
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from toolkit.config_modules import ModelConfig, NetworkConfig
from toolkit.lora_special import LoRASpecialNetwork
from extensions_built_in.diffusion_models.krea2.krea2 import Krea2Model


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def infer_lora_rank(path):
    with safe_open(Path(path), framework="pt", device="cpu") as checkpoint:
        key = next((name for name in checkpoint.keys() if name.endswith(".lora_A.weight")), None)
        if key is None:
            raise RuntimeError(f"No LoRA A tensor was found in {path}")
        return int(checkpoint.get_slice(key).get_shape()[0])


def capture_base_parameter_samples(transformer, sample_count=6, sample_length=2048):
    candidates = [
        (name, parameter)
        for name, parameter in transformer.named_parameters()
        if "lora_" not in name.lower() and ".lora" not in name.lower()
    ]
    if not candidates:
        raise RuntimeError("No non-LoRA transformer parameters were found for the non-destructive audit.")
    indices = sorted(set(round(index * (len(candidates) - 1) / max(sample_count - 1, 1)) for index in range(min(sample_count, len(candidates)))))
    return {
        candidates[index][0]: candidates[index][1].detach().reshape(-1)[:sample_length].float().cpu().clone()
        for index in indices
    }


def compare_base_parameter_samples(transformer, samples):
    parameter_map = dict(transformer.named_parameters())
    records = []
    unchanged = True
    for name, before in samples.items():
        if name not in parameter_map:
            raise RuntimeError(f"A sampled base parameter disappeared: {name}")
        after = parameter_map[name].detach().reshape(-1)[:before.numel()].float().cpu()
        equal = bool(torch.equal(before, after))
        unchanged = unchanged and equal
        records.append({
            "name": name,
            "unchanged": equal,
            "maximum_difference": float((before - after).abs().max().item()),
        })
    return unchanged, records


class MultiAdapterController:
    def __init__(self, records):
        self.records = records
        self.can_merge_in = False
        self.is_merged_in = False
        self.is_active = True
        self.training = False
        self._multiplier = 1.0

    @property
    def multiplier(self):
        return self._multiplier

    @multiplier.setter
    def multiplier(self, value):
        self._multiplier = float(value)
        self._apply()

    def _apply(self):
        for record in self.records:
            network = record["network"]
            effective = float(record["scale"]) * self._multiplier
            network.multiplier = effective
            network.is_active = effective != 0.0
            network.can_merge_in = False
            network.is_merged_in = False
            network._update_torch_multiplier()
            multiplier_tensor = network.torch_multiplier.detach()
            if not torch.isfinite(multiplier_tensor.float()).all():
                raise RuntimeError(f"Adapter multiplier is non-finite: {record['name']}")
            expected_value = torch.tensor(
                effective,
                dtype=multiplier_tensor.dtype,
                device=multiplier_tensor.device,
            ).float()
            received_values = multiplier_tensor.float()
            if not torch.all(received_values == expected_value):
                raise RuntimeError(
                    f"Adapter multiplier mismatch for {record['name']}: "
                    f"expected {expected_value.item()}, received {received_values.cpu().tolist()}"
                )

    def set_scales(self, scales):
        unknown = set(scales) - {record["name"] for record in self.records}
        if unknown:
            raise RuntimeError(f"Unknown adapter names: {sorted(unknown)}")
        for record in self.records:
            if record["name"] in scales:
                record["scale"] = float(scales[record["name"]])
        self._apply()

    def set_global_multiplier(self, value):
        self.multiplier = value

    def eval(self):
        self.training = False
        for record in self.records:
            record["network"].eval()
        return self

    def train(self, mode=True):
        self.training = bool(mode)
        for record in self.records:
            record["network"].train(mode)
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def load_runtime(inference_assets, adapter_specs, dtype="bf16", max_text_length=512):
    turbo = inference_assets["inference_model"]
    model_configuration = ModelConfig(
        name_or_path=str(Path(turbo["checkpoint_path"]).parent),
        arch="krea2",
        dtype=dtype,
        vae_dtype=dtype,
        te_dtype=dtype,
        quantize=False,
        quantize_te=False,
        low_vram=False,
        layer_offloading=False,
        split_model_over_gpus=False,
        compile=False,
        assistant_lora_path=None,
        inference_lora_path=None,
        model_kwargs={
            "checkpoint_filename": turbo["checkpoint_filename"],
            "text_encoder_path": inference_assets["text_encoder"]["local_directory"],
            "vae_path": inference_assets["vae"]["local_directory"],
            "max_text_length": max_text_length,
        },
    )
    model = Krea2Model(device="cuda:0", model_config=model_configuration, dtype=dtype)
    model.load_model()
    transformer = getattr(model, "unet", None)
    if transformer is None:
        transformer = model.get_model_to_train()
    if isinstance(transformer, (list, tuple)):
        if len(transformer) != 1:
            raise RuntimeError(f"Unexpected transformer count: {len(transformer)}")
        transformer = transformer[0]
    if transformer is None:
        raise RuntimeError("The Krea 2 transformer could not be resolved.")
    transformer.eval()
    target_modules = list(model.target_lora_modules)
    if target_modules != ["SingleStreamDiT"]:
        raise RuntimeError(f"Unexpected Krea 2 LoRA target modules: {target_modules}")
    LoRASpecialNetwork.LORA_PREFIX_UNET = "lora_transformer"
    records = []
    names = set()
    for adapter_spec in adapter_specs:
        name = adapter_spec["name"]
        if name in names:
            raise RuntimeError(f"Duplicate adapter name: {name}")
        names.add(name)
        adapter_path = Path(adapter_spec["path"])
        if not adapter_path.is_file():
            raise RuntimeError(f"Adapter file is missing: {adapter_path}")
        rank = infer_lora_rank(adapter_path)
        alpha_value = adapter_spec.get("alpha")
        alpha = rank if alpha_value is None else int(alpha_value)
        network_configuration = NetworkConfig(type="lora", linear=rank, linear_alpha=alpha, transformer_only=True)
        network = LoRASpecialNetwork(
            text_encoder=None,
            unet=transformer,
            lora_dim=rank,
            multiplier=float(adapter_spec.get("scale", 1.0)),
            alpha=alpha,
            train_unet=True,
            train_text_encoder=False,
            network_config=network_configuration,
            network_type="lora",
            transformer_only=True,
            is_transformer=True,
            target_lin_modules=target_modules,
            base_model=model,
        )
        network.adapter_name = name
        network.apply_to(None, transformer, apply_text_encoder=False, apply_unet=True)
        network.force_to(model.device_torch, dtype=model.torch_dtype)
        network.eval()
        network.can_merge_in = False
        network.is_merged_in = False
        state_dict = load_file(str(adapter_path))
        state_dict = model.convert_lora_weights_before_load(state_dict)
        network.load_weights(state_dict)
        nonfinite_count = sum(int((~torch.isfinite(parameter.detach().float())).sum().item()) for parameter in network.parameters())
        if nonfinite_count != 0:
            raise RuntimeError(f"Adapter contains non-finite runtime parameters: {adapter_path}")
        records.append({
            "name": name,
            "path": str(adapter_path),
            "sha256": sha256_file(adapter_path),
            "scale": float(adapter_spec.get("scale", 1.0)),
            "rank": rank,
            "alpha": alpha,
            "parameter_count": sum(int(parameter.numel()) for parameter in network.parameters()),
            "network": network,
        })
    controller = MultiAdapterController(records)
    controller.set_scales({record["name"]: record["scale"] for record in records})
    model.network = controller
    return model, transformer, controller, records


def unload_runtime(model, transformer, controller, records):
    controller.set_scales({record["name"]: 0.0 for record in records})
    del records
    del controller
    del transformer
    del model
    gc.collect()
    torch.cuda.empty_cache()
"""
helper_path.parent.mkdir(parents=True, exist_ok=True)
helper_path.write_text(helper_source, encoding="utf-8")
print(f"Inference helper module: {helper_path}")