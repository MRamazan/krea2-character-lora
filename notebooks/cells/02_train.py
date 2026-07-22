# ============================================================
# Training
# ============================================================

from krea2_character_lora import TrainingConfig

PROJECT_NAME = "krea2_character_lora"
RUN_NAME = "character_v1"
TRAINING_STEPS = 2000
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 1
TRAINING_RESOLUTIONS = (768, 1024)
LORA_RANK = 32
LORA_ALPHA = 32
SAVE_EVERY = 200
MAX_CHECKPOINTS_TO_KEEP = 10
DATASET_REPEATS = 1
CAPTION_DROPOUT_RATE = 0.0
TOKEN_DROPOUT_RATE = 0.0
SHUFFLE_TOKENS = False
KEEP_TOKENS = 1
FLIP_X = False
TRAINING_DTYPE = "bf16"
GENERATE_TRAINING_SAMPLES = False
TRAINING_SAMPLE_EVERY = 200
RUN_SMOKE_TEST = True
SMOKE_TEST_STEPS = 3
RUN_PRODUCTION_TRAINING = True
RESUME_MODE = "auto"

training_config = TrainingConfig(
    project_name=PROJECT_NAME,
    run_name=RUN_NAME,
    training_steps=TRAINING_STEPS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    batch_size=BATCH_SIZE,
    gradient_accumulation=GRADIENT_ACCUMULATION,
    resolutions=TRAINING_RESOLUTIONS,
    lora_rank=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    save_every=SAVE_EVERY,
    max_checkpoints_to_keep=MAX_CHECKPOINTS_TO_KEEP,
    dataset_repeats=DATASET_REPEATS,
    caption_dropout_rate=CAPTION_DROPOUT_RATE,
    token_dropout_rate=TOKEN_DROPOUT_RATE,
    shuffle_tokens=SHUFFLE_TOKENS,
    keep_tokens=KEEP_TOKENS,
    flip_x=FLIP_X,
    training_dtype=TRAINING_DTYPE,
    generate_training_samples=GENERATE_TRAINING_SAMPLES,
    training_sample_every=TRAINING_SAMPLE_EVERY,
)

pipeline.preview_training(
    dataset=dataset,
    config=training_config,
)

training_run = pipeline.train(
    dataset=dataset,
    config=training_config,
    run_smoke_test=RUN_SMOKE_TEST,
    smoke_test_steps=SMOKE_TEST_STEPS,
    run_production=RUN_PRODUCTION_TRAINING,
    resume=RESUME_MODE,
)

training_run.display_summary()
training_run.show_checkpoints()
