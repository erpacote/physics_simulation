import numpy as np
import matplotlib.pyplot as plt

from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.optimizers import *
from tensorflow.python.keras.callbacks import *
import tensorflow.python.keras.backend as K


def get_2D_path(steps, trig_coeffs, coeffs_x, coeffs_y):
    t = np.linspace(0.0, 10.0, steps)

    x = trig_coeffs[0] * np.sin(t)
    y = trig_coeffs[0] * np.cos(t)
    x = x + sum([coeff * t ** ii for ii, coeff in enumerate(coeffs_x)])
    y = y + sum([coeff * t ** ii for ii, coeff in enumerate(coeffs_y)])

    return x, y


def get_input_output_example(cfg, plot=False):
    x, y = get_2D_path(cfg.n_time_steps + 1, [1.0, 1.0], [0.0, 4.0, -0.2, -0.02],
                       [0.0, -2.0, 0.5, 0.04])

    wpts = np.array([(x[wp_t], y[wp_t]) for wp_t
                     in range(cfg.n_time_steps - 1, 0, -int(cfg.n_time_steps / cfg.n_waypoints))])

    new_output = np.array([x[1:], y[1:]])  # first sample is where we start
    new_output = new_output.swapaxes(1, 0)
    new_output = np.expand_dims(new_output, 0)

    new_input = np.hstack([np.array([x[0], y[0]]), wpts.flatten()])
    new_input = np.expand_dims(new_input, axis=0)
    new_input = np.stack([new_input] * cfg.n_time_steps, axis=1)

    max_val = np.max([x, y])
    min_val = np.min([x, y])
    scale = max_val - min_val
    bias = (max_val + min_val) / 2

    if plot:
        print('wpts')
        print(wpts.shape)
        print(wpts[:3])

        print('output', new_output.shape)
        print(new_output[0, :5, :])

        plt.figure()
        plt.plot(new_output[0, :, 0], new_output[0, :, 1])
        plt.scatter(wpts[:, 0], wpts[:, 1], marker='x')

        print('input')
        print(new_input.shape)
        print(new_input[0, 1, :5])
        print(new_input[0, 2, :5])
        print(new_input[0, 3, :5])

    return new_input, new_output, wpts, scale, bias


from dataclasses import dataclass
from typing import List


@dataclass
class Config:
    repetitions: str
    lstm_units: List
    dense_units: List
    loss: str
    start_lr: float
    reduce_rl_patience: int
    cooldown: int
    stop_patience: int
    n_time_steps: int
    n_waypoints: int
    kernel_initializer: str = 'glorot_uniform'


def plot_fig(wpts, X, y, model, hist, cfg, scale, bias, lr_log=None):
    prediction = scale_to_original(model.predict(X[:1, :, :]), scale, bias)
    actual = scale_to_original(y, scale, bias)
    start_point = scale_to_original(X[:1, :1, :2], scale, bias)

    print(cfg)
    print('prediction', prediction.shape)

    losses = hist.history['loss']

    f = plt.figure(figsize=(10, 10))
    ax = f.add_subplot(121)
    ax.plot(prediction[0, :, 0], prediction[0, :, 1], '-x', label='pred')
    ax.plot(actual[0, :, 0], actual[0, :, 1], '-x', color='r', label='actual')
    ax.scatter(wpts[:, 0], wpts[:, 1], marker='o', label='wpts')
    ax.scatter(start_point[0, 0, 0], start_point[0, 0, 1], marker='o', label='start')
    ax.grid()
    ax.legend()

    ax = f.add_subplot(222)
    ax.plot(losses)
    axes = ax.axis()
    #     min_ = min([0.13, min(losses[n//2:])*0.98])
    #     max_ = max(losses[n//2:])*1.02
    #     ax.axis([*axes[:2], min_, max_])
    ax.grid()

    if lr_log:
        ax = f.add_subplot(224, sharex=ax)
        ax.plot(lr_log)
        ax.grid()


def get_model(cfg):
    model = Sequential()
    model.add(LSTM(units=cfg.lstm_units[0],
                   input_shape=(cfg.n_time_steps, cfg.n_waypoints * 2 + 2),
                   return_sequences=True,
                   kernel_initializer=cfg.kernel_initializer))
    for n_cells in cfg.lstm_units[1:]:
        model.add(LSTM(units=n_cells,
                       return_sequences=True,
                       kernel_initializer=cfg.kernel_initializer))
    for n_cells in cfg.dense_units:
        model.add(TimeDistributed(Dense(n_cells,
                                        activation='relu',
                                        kernel_initializer=cfg.kernel_initializer)))
    model.add(TimeDistributed(Dense(2, kernel_initializer=cfg.kernel_initializer)))
    # model.add((Dense(50)))
    # model.add((Dense(2)))

    model.compile(optimizer=Adam(lr=cfg.start_lr), loss=cfg.loss, metrics=[cfg.loss])
    # model.summary()

    return model


def scale_for_training(data, scale, bias):
    return (data - bias) / scale


def scale_to_original(scaled_data, scale, bias):
    return scaled_data * scale + bias


def get_repeated_batch(n_repetitions, one_input, one_output, scale, bias):
    input_batch = np.concatenate([one_input] * n_repetitions, axis=0)
    output_batch = np.concatenate([one_output] * n_repetitions, axis=0)

    X = scale_for_training(input_batch, scale, bias)
    y = scale_for_training(output_batch, scale, bias)

    print('X', X.shape, 'y', y.shape)

    return X, y


class InformativeCallback(Callback):
    def __init__(self, lr_log, verbose=0):
        super().__init__()
        self.lr_log = lr_log

    def on_epoch_end(self, epoch, logs=None):
        lr = K.eval(self.model.optimizer.lr)
        self.lr_log.append(lr)
        n = self.params['epochs'] // 10
        print(f'\rEpoch {(epoch + 1):3} finished, lr: {lr:.3e}, loss: {logs["loss"]:6.4f} ...',
              end='')
        if (epoch + 1) % n == 0:
            print('')  # Leave a line fixed


def get_callbacks(lr_log, cfg):
    callbacks = [
        ReduceLROnPlateau(monitor='loss',
                          factor=0.5,
                          patience=cfg.reduce_rl_patience,
                          cooldown=cfg.cooldown,
                          verbose=0,
                          min_lr=1.0e-12),
        ModelCheckpoint(filepath='(auto)' + str(cfg) + '-best.hdf5',
                        monitor='loss',
                        save_best_only=True),
        InformativeCallback(lr_log, 0),
        EarlyStopping(monitor='loss',
                      min_delta=0,
                      patience=cfg.stop_patience,
                      verbose=1,
                      baseline=None)
    ]
    return callbacks