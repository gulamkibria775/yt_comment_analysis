import mlflow
from mlflow.exceptions import RestException

# === Set MLflow Tracking URI to your DagsHub repository ===
mlflow.set_tracking_uri("https://dagshub.com/gulamkibria775/yt_comment_analysis.mlflow")

# === Delete model versions and registered model from DagsHub ===
def delete_registered_model(model_name: str):
    client = mlflow.MlflowClient()

    try:
        # Step 1: Delete all model versions
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            print(f"No versions found for model '{model_name}'")

        for mv in versions:
            version = mv.version
            client.delete_model_version(name=model_name, version=version)
            print(f"✅ Deleted version {version} of model '{model_name}'")

        # Step 2: Delete the registered model
        client.delete_registered_model(name=model_name)
        print(f"✅ Successfully deleted registered model '{model_name}' from DagsHub")

    except RestException as e:
        print(f"❌ MLflow REST Exception: {e}")
    except Exception as e:
        print(f"❌ Unexpected error occurred: {e}")

if __name__ == "__main__":
    # Replace with the exact name of the model you registered in DagsHub
    model_name_to_delete = "yt_chrome_plugin_model"
    delete_registered_model(model_name_to_delete)
