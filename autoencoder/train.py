from tuner import create_tuner
from model import build_model
import keras

from data import X_train, X_valid, X_test

tuner = create_tuner(6, overwrite=False)
best_hps = tuner.get_best_hyperparameters(1)

ae = build_model(best_hps[0])

callbacks = [
    keras.callbacks.EarlyStopping(monitor="loss", patience=10, restore_best_weights=True, min_delta=0.001)
]

ae.fit(
    np.concatenate([X_train, X_valid, X_test]), batch_size=1024, epochs=100, callbacks=callbacks
)

ae.save_weights(f"./weights/final/final")