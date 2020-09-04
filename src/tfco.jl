ENV["PYTHON"] = "/opt/rh/rh-python36/root/usr/bin/python"

using PyCall

py"""
import numpy as np
import tensorflow as tf
import tensorflow_constrained_optimization as tfco
import time


def batch_provider(data, num_epochs=10, shuffle=True, batch_size=32):
    def make_batch():
        dataset = tf.data.Dataset.from_tensor_slices(data)
        if shuffle:
            dataset = dataset.shuffle(1000)
        dataset = dataset.batch(batch_size).repeat(num_epochs)
        return dataset
    return make_batch


def prepare_outputs(eval_output_train, eval_output_test, elapsed_time):
    eval_output_train.pop('loss', None)
    eval_output_train.pop('global_step', None)
    eval_output_test.pop('loss', None)
    eval_output_test.pop('global_step', None)

    output = {
        'elapsed time': elapsed_time,
        'train': {
            'fpr': [float(key) for key in eval_output_train.keys()],
            'tpr': [float(val) for val in eval_output_train.values()],
        },
        'test': {
            'fpr': [float(key) for key in eval_output_test.keys()],
            'tpr': [float(val) for val in eval_output_test.values()],
        },
    }
    return output


def build_model_mnist():
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(20, kernel_size=(5,5), activation='relu', input_shape=(28,28,1), strides=(1, 1)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(50, kernel_size=(5,5), activation='relu', strides=(1, 1)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(500),
        tf.keras.layers.Dense(1),
    ])


def build_model_cifar():
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu', input_shape=(32,32,3), padding="valid"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(128, kernel_size=3, activation='relu', padding="valid"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(128, kernel_size=3, activation='relu', padding="valid"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1),
    ])


def build_model_svhn2():
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu', input_shape=(32,32,3), padding="valid"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(128, kernel_size=3, activation='relu', padding="valid"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(128, kernel_size=3, activation='relu', padding="valid"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1),
    ])


def make_model_fn(build_model, fpr, optimizer, steplength, fpr_rates):
    def model_fn(features, labels, mode):
        model = build_model()
        scores = model(features)

        # Baseline cross-entropy loss.
        loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        loss = loss_fn(labels, scores)
        train_op = None

        # output predictions
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=scores)

        # Set up precision at recall optimization problem.
        if mode == tf.estimator.ModeKeys.TRAIN:
            # Create optimization problem
            context = tfco.rate_context(scores, labels)
            problem = tfco.RateMinimizationProblem(
                tfco.false_negative_rate(context),
                [tfco.false_positive_rate(context) <= fpr]
            )

            lag_loss, update_ops_fn, multipliers = tfco.create_lagrangian_loss(problem)

            # Set up optimizer and the list of variables to optimize the loss.
            opt = optimizer(learning_rate=steplength)
            opt.iterations = tf.compat.v1.train.get_or_create_global_step()

            # Get minimize op and group with update_ops.
            var_list = (model.trainable_weights +
                        problem.trainable_variables + [multipliers])
            minimize_op = opt.get_updates(lag_loss(), var_list)
            update_ops = update_ops_fn()
            train_op = tf.group(*update_ops, minimize_op)

            return tf.estimator.EstimatorSpec(
                mode,
                loss=loss,
                train_op=train_op
            )

        if mode == tf.estimator.ModeKeys.EVAL:
            # Define the metrics:
            metric_dict = {}
            for rate in fpr_rates:
                metric = tf.keras.metrics.SensitivityAtSpecificity(1 - rate)
                metric.update_state(labels, tf.sigmoid(scores))
                metric_dict[f'{rate}'] = metric

            return tf.estimator.EstimatorSpec(
                mode,
                loss=loss,
                eval_metric_ops=metric_dict,
            )
    return model_fn


def train_model(build_model, x_train, y_train, x_test, y_test, args):
    fpr = args["fpr"]
    batchsize = args["batchsize"]
    epochs = args["epochs"]
    steplength = args["steplength"]
    optimizer = tf.keras.optimizers.SGD
    fpr_rates = np.logspace(-4, 0, num=300)

    # create model
    classifier = tf.estimator.Estimator(
        make_model_fn(build_model, fpr, optimizer, steplength, fpr_rates),
    )

    # train model
    make_batch = batch_provider((x_train, y_train), num_epochs=epochs, shuffle=True, batch_size=batchsize)

    start_time = time.time()
    classifier.train(make_batch)
    elapsed_time = time.time() - start_time

    # create data iterators for evaluation
    eval_data_train = batch_provider((x_train, y_train), num_epochs=1, shuffle=False)
    eval_data_test = batch_provider((x_test, y_test), num_epochs=1, shuffle=False)

    # save results
    return prepare_outputs(
        classifier.evaluate(eval_data_train),
        classifier.evaluate(eval_data_test),
        elapsed_time,
    )
"""

build_network_tfco(::Type{<:AbstractMNIST}) = py"build_model_mnist"
build_network_tfco(::Type{<:AbstractCIFAR}) = py"build_model_cifar"
build_network_tfco(::Type{<:AbstractSVHN2}) = py"build_model_svhn2"

function reshape_for_python(x, y)
    d = ndims(x)
    return Float32.(permutedims(x, vcat(d, 1:(d-1)))), Float32.(vec(y))
end
