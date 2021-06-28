from dotenv import find_dotenv, load_dotenv
from comet_ml import Experiment
import os


def create_experiment():
    load_dotenv(find_dotenv())

    experiment = Experiment(
        api_key=os.environ["COMET_API_KEY"],
        project_name="interpretable-trading",
        workspace="g8a9",
    )
    return experiment