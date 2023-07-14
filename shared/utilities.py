import tensorflow as tf

SAT1_STAGES_ACCURACY = ['encoding', 'decision', 'confirmation', 'response']
SAT1_STAGES_SPEED = ['encoding', 'decision', 'response']
SAT2_STAGES_ACCURACY = ['encoding', 'decision', 'confirmation', 'response']
SAT2_STAGES_SPEED = ['encoding', 'decision', 'response']
AR_STAGES = ['encoding', 'familiarity', 'memory', 'decision', 'response']

earlyStopping_cb = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=2,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=True,
    start_from_epoch=0,
)
