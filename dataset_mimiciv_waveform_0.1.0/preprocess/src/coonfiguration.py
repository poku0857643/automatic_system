import os
from logging import getLogger

from src.constants import CONSTANTS, PLATFORM_ENUM

logger = getLogger(__name__)

class PlatformConfigurations:
    platform = os.getenv("PLATFORM_ENUM", PLATFORM_ENUM.DOCKER.value)
    if not PLATFORM_ENUM.has_value(platform):
        raise ValueError(f"PLATFORM must be one of {[v.value for v in PLATFORM_ENUM.__members__.values()]}")

class PreprocessConfigurations:
    # File for training and testing data
    train_files = [
        "sbp_dbp_train_1.csv",
        "sbp_dbp_train_2.csv",
        "sbp_dbp_train_3.csv",
        "sbp_dbp_train_4.csv",
        "sbp_dbp_train_5.csv",
    ]
    test_files = ["sbp_dbp_test.csv"]

    # Physiological_features classes for prompt generation
    physiological_features = {
        "CO_features": [0.1, 0.2, 0.3],
        "PTT_features": [0.1, 0.2, 0.3],
        "PR_features": [0.1, 0.2, 0.3],
    }

    @classmethod
    def generate_prompt(cls, features):
        co_str = ", ".join(map(str, features["CO_features"]))
        ptt_str = ", ".join(map(str, features["PTT_features"]))
        pr_str = ", ".join(map(str, features["PR_features"]))

        return (
            f"The physiological features are CO features: [{co_str}],"
            f"PTT features: [{ptt_str}], PR features: [{pr_str}], "
            f"Based on these data, predict the systolic blood pressure (SBP),"
            f"and diastolic blood pressure (DBP) values."
        )

class ModelConfigurations:
    # model-specific configurations for SBP and DBP regression
    input_dim = len(PreprocessConfigurations.physiological_features)
    output_dim = 2 # predicting SBP and DBP as separate values
    loss_function = "MSELoss" # Mean Squared Error for regression
    optimizer = "Adam"
    learning_rate = 1e-3
    epochs = 50

    # Logging model configurations
    #classmethod
    def log_configurations(cls):
        logger.info(f"Model Configurations: Input Dimension={cls.input_dim},"
                    f"Output Dimension={cls.output_dim}, Loss Function={cls.loss_function},"
                    f"Optimizer={cls.optimizer}, Learning Rate={cls.learning_rate}, Epochs={cls.epochs}")

# Example usage
if __name__ == "__main__":
    #Generate a sample prompt for logging
    prompt = PreprocessConfigurations.generate_prompt(PreprocessConfigurations.physiological_features)
    logger.info(f"Generated Prompt: {prompt}")

    # Log all configurations
    logger.info(f"{PlatformConfigurations.__name__}: {PlatformConfigurations.__dict__}")
    logger.info(f"{ModelConfigurations.__name__}: {ModelConfigurations.__dict__}")
    ModelConfigurations.log_configurations()

