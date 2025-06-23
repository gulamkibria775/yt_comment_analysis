import json
import mlflow
import logging

# Set up MLflow tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5001")

# Logging configuration
logger = logging.getLogger('model_registration')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('model_registration_errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logger.debug(f'Model info loaded from {file_path}')
        return model_info
    except FileNotFoundError:
        logger.error(f'File not found: {file_path}')
        raise
    except Exception as e:
        logger.error(f'Unexpected error while loading model info: {e}')
        raise

def register_model(model_name: str, model_info: dict):
    """Register the model to MLflow Model Registry and transition to staging."""
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        logger.debug(f"Registering model from URI: {model_uri}")

        # Register model
        model_version = mlflow.register_model(model_uri, model_name)
        logger.info(f"Registered model {model_name} version {model_version.version}")

        # Optional: Transition to Staging
        try:
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage="Staging"
            )
            logger.info(f"Transitioned model {model_name} version {model_version.version} to Staging")
        except Exception as stage_error:
            logger.warning(f"Stage transition failed or skipped: {stage_error}")

    except Exception as e:
        logger.error(f"Error during model registration: {e}")
        raise

def main():
    try:
        model_info_path = 'experiment_info.json'  # Path to JSON with run_id and model_path
        model_info = load_model_info(model_info_path)

        model_name = "yt_chrome_plugin_model"  # Name of your registered model
        register_model(model_name, model_info)

    except Exception as e:
        logger.error(f"Failed to complete model registration: {e}")
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
