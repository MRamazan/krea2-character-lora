from dataclasses import dataclass

TRAINING_MODEL_REPOSITORY = "krea/Krea-2-Raw"
TRAINING_CHECKPOINT_FILENAME = "raw.safetensors"
INFERENCE_MODEL_REPOSITORY = "krea/Krea-2-Turbo"
INFERENCE_CHECKPOINT_FILENAME = "turbo.safetensors"
TEXT_ENCODER_REPOSITORY = "Qwen/Qwen3-VL-4B-Instruct"

CUSTOM_VAE_REPOSITORY = "artsyww/KREA2REALVAE"
VAE_CONFIG_REPOSITORY = "Qwen/Qwen-Image"
VAE_CONFIG_SUBFOLDER = "vae"
VAE_STATE_DICT_PREFIX = "vae."
VAE_ARCHITECTURE = "AutoencoderKLQwenImage"

VAE_FORMAT_DIFFUSERS = "diffusers_autoencoder_kl_qwen_image"
VAE_FORMAT_ORIGINAL = "qwen_image_original_vae"
VAE_FORMAT_UNKNOWN = "unknown"
VAE_CONVERSION_IDENTITY = "identity"
VAE_CONVERSION_ORIGINAL_TO_DIFFUSERS = "wan_qwen_original_to_diffusers"

AI_TOOLKIT_REPOSITORY = "https://github.com/ostris/ai-toolkit.git"
AI_TOOLKIT_REVISION = "main"
VIRTUALENV_VERSION = "21.6.1"
TORCH_VERSION = "2.9.1"
TORCHVISION_VERSION = "0.24.1"
TORCHAUDIO_VERSION = "2.9.1"
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu128"

MODEL_ARCHITECTURE = "krea2"
TARGET_LORA_MODULES = ["SingleStreamDiT"]
PRIMARY_ADAPTER_NAME = "character_adapter"
MAX_TEXT_LENGTH = 512

MINIMUM_TRAINING_DISK_BYTES = 42 * 1024**3
MINIMUM_INFERENCE_DISK_BYTES = 27 * 1024**3

ACCEPTED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
KNOWN_UNSUPPORTED_IMAGE_EXTENSIONS = (".webp", ".bmp", ".tif", ".tiff", ".gif")
CAPTION_EXTENSION = ".txt"

BUNDLE_TYPE = "krea2_character_lora_evaluation_bundle"
BUNDLE_FORMAT_VERSION = 1
BUNDLE_MANIFEST_NAME = "bundle_manifest.json"
MAX_BUNDLE_FILES = 5000
MAX_BUNDLE_UNCOMPRESSED_BYTES = 8 * 1024**3
MAX_LORA_FILE_BYTES = 1024**3
PACKAGE_DISTRIBUTION = "krea2-character-lora"
FALLBACK_PACKAGE_VERSION = "0.1.0"


@dataclass(frozen=True, slots=True)
class RepositoryAsset:
    repository: str
    filename: str | None = None
    subfolder: str | None = None
