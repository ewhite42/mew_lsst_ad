
import keras
from tuner import create_tuner

from data import X_train, X_valid

latent_components = [2, 4, 6, 8, 10]

for i in latent_components:
    # Bayesian optimization search.

    print(f"Starting latent dim {i}")
    print("-------")
    tuner = create_tuner(ld = i, overwrite=True)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True, min_delta=0.01)
    ]
    tuner.search(X_train, validation_data=(X_valid,), batch_size=1024, epochs=10, callbacks=callbacks)
    print("------")

for i in latent_components:
    # Train best hyperparam model for each latent space component.
    
    print(f"Starting latent dimensions {i}")
    print("-------")
    tuner = create_tuner(i, overwrite=False)
    best_hps = tuner.get_best_hyperparameters(1)

    ae = build_model(best_hps[0])

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, min_delta=0.001)
    ]
    ae.fit(
        X_train, validation_data=(X_valid,), batch_size=1024, epochs=100, callbacks=callbacks
    )
    ae.save_weights(f"./weights/ld_{i}/best")

    print(f"Finished latent dimensions {i}")
    print("------")