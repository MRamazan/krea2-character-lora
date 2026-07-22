# ============================================================
# Evaluation and export
# ============================================================

from krea2_character_lora import EvaluationConfig

TRIGGER_WORD = imported_run.trigger_word

PROMPTS = [
    f"{TRIGGER_WORD} is a woman in a fully clothed professional studio portrait, "
    "wearing a tailored blazer, soft key lighting, sharp facial detail",
    f"{TRIGGER_WORD} is a woman standing outdoors in natural daylight, "
    "wearing a casual jacket and jeans, fully clothed, relaxed pose",
    f"{TRIGGER_WORD} is a woman taking a mirror selfie in a modern room, "
    "holding a phone, wearing a fully clothed everyday outfit",
    f"{TRIGGER_WORD} is a woman seated and taking a mirror selfie, "
    "wearing a long-sleeve top and trousers, natural indoor lighting",
    f"{TRIGGER_WORD} is a woman taking an indoor selfie near a window, "
    "wearing a knitted sweater, warm ambient light, clear facial features",
    f"{TRIGGER_WORD} is a woman in a lifestyle photograph at a cafe, "
    "wearing a coat and scarf, fully clothed, candid expression",
]
SEEDS = [42, 12345, 987654321]
WIDTH = 1024
HEIGHT = 1024
INFERENCE_STEPS = 8
GUIDANCE_SCALE = 0.0
NEGATIVE_PROMPT = ""
CHECKPOINT_MODE = "auto"
MAXIMUM_CHECKPOINTS = 8
MANUAL_CHECKPOINT_STEPS = []
PRIMARY_ADAPTER_SCALE = 1.0
SCALE_SWEEP = [0.6, 0.8, 1.0]
COMPARE_BASE_MODEL = True
INCLUDE_BASE_IN_CHECKPOINT_GRID = True
RUN_CHECKPOINT_SWEEP = True
RUN_SCALE_SWEEP = True
INCLUDE_ALL_CHECKPOINTS_IN_EXPORT = False
DOWNLOAD_EXPORTS = False

evaluation_config = EvaluationConfig(
    prompts=PROMPTS,
    seeds=SEEDS,
    width=WIDTH,
    height=HEIGHT,
    inference_steps=INFERENCE_STEPS,
    guidance_scale=GUIDANCE_SCALE,
    negative_prompt=NEGATIVE_PROMPT,
    checkpoint_mode=CHECKPOINT_MODE,
    maximum_checkpoints=MAXIMUM_CHECKPOINTS,
    manual_checkpoint_steps=MANUAL_CHECKPOINT_STEPS,
    primary_adapter_scale=PRIMARY_ADAPTER_SCALE,
    scale_sweep=SCALE_SWEEP,
    compare_base_model=COMPARE_BASE_MODEL,
    include_base_in_checkpoint_grid=INCLUDE_BASE_IN_CHECKPOINT_GRID,
    run_checkpoint_sweep=RUN_CHECKPOINT_SWEEP,
    run_scale_sweep=RUN_SCALE_SWEEP,
)

pipeline.prepare_evaluation_assets()

evaluation = pipeline.evaluate(
    run=imported_run,
    config=evaluation_config,
)

evaluation.show_base_comparison()
evaluation.show_checkpoint_grid()
evaluation.show_scale_grid()
evaluation.show_summary()

exports = evaluation.export(
    include_selected_lora=True,
    include_all_checkpoints=INCLUDE_ALL_CHECKPOINTS_IN_EXPORT,
    include_images=True,
    include_logs=True,
    include_manifests=True,
)

exports.display()

if DOWNLOAD_EXPORTS:
    exports.download()
