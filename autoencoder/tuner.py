import kerastuner as kt
from model import latent_model

def create_tuner(ld, overwrite=False):
    tuner = kt.BayesianOptimization(
        hypermodel=latent_model(ld=ld),
        objective="val_loss",
        max_trials=5,
        executions_per_trial=1,
        overwrite=overwrite,
        directory=f"./tuner/ld_{ld}",
        project_name="tuner_results",
    )
    return tuner
