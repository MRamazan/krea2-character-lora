class Krea2CharacterLoraError(RuntimeError):
    pass


class ConfigurationError(Krea2CharacterLoraError):
    pass


class EnvironmentPreparationError(Krea2CharacterLoraError):
    pass


class AssetError(Krea2CharacterLoraError):
    pass


class VaeValidationError(AssetError):
    pass


class DatasetError(Krea2CharacterLoraError):
    pass


class TrainingError(Krea2CharacterLoraError):
    pass


class CheckpointError(Krea2CharacterLoraError):
    pass


class EvaluationError(Krea2CharacterLoraError):
    pass


class ExportError(Krea2CharacterLoraError):
    pass
