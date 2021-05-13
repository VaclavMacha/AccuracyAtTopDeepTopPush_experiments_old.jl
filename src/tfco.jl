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

def build_model_imagenet():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(1, input_shape=(10240,)),
    ])

def build_model_molecules():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(50, input_shape=(100,), activation='sigmoid'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(25, activation='sigmoid'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(1),
    ])


def make_model_fn(build_model, fpr, optimizer, steplength):
    def model_fn(features, labels, mode):
        model = build_model()
        scores = model(features)

        # output predictions
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=scores)

        # Baseline cross-entropy loss.
        loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        loss = loss_fn(labels, scores)
        train_op = None

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
    return model_fn


def train_model(build_model, x_train, y_train, x_test, y_test, args, model_dir):
    # create model
    classifier = tf.estimator.Estimator(
        model_fn = make_model_fn(build_model, args["fpr"], args["optimiser"], args["steplength"]),
        config=tf.estimator.RunConfig(tf_random_seed=args["seed"]),
        model_dir = model_dir,
    )

    # train model
    make_batch = batch_provider((x_train, y_train), num_epochs=args["epochs"], shuffle=True, batch_size=args["batchsize"])

    start_time = time.time()
    classifier.train(make_batch)
    elapsed_time = time.time() - start_time

    # compute scores
    eval_data_train = batch_provider((x_train, y_train), num_epochs=1, shuffle=False)
    eval_data_test = batch_provider((x_test, y_test), num_epochs=1, shuffle=False)

    eval_train = classifier.evaluate(eval_data_train)
    eval_test = classifier.evaluate(eval_data_test)

    s_train = np.array(list(classifier.predict(input_fn=eval_data_train)))
    s_test = np.array(list(classifier.predict(input_fn=eval_data_test)))

    return {
        "s_train": s_train,
        "loss_train": eval_train['loss'],
        "s_test": s_test,
        "loss_test": eval_test['loss'],
        "tm": elapsed_time,
        "iters": eval_train['global_step'],
    }
"""

build_network_tfco(::Type{<:AbstractMNIST}) = py"build_model_mnist"
build_network_tfco(::Type{<:AbstractCIFAR}) = py"build_model_cifar"
build_network_tfco(::Type{<:AbstractSVHN2}) = py"build_model_svhn2"
build_network_tfco(::Type{<:Molecules}) = py"build_model_molecules"
build_network_tfco(::Type{<:ImageNet}) = py"build_model_cifar"
build_network_tfco(::Type{<:ImageNetPrep}) = py"build_model_imagenet"


function reshape_for_python(x, y)
    d = ndims(x)
    return Float32.(permutedims(x, vcat(d, 1:(d-1)))), Float32.(vec(y))
end

function run_simulations_tfco(Dataset_Settings, Train_Settings, Model_Settings)
    model_dir = mktempdir("."; cleanup=false)
    for dataset_settings in dict_list_simple(Dataset_Settings)
        @unpack dataset, posclass = dataset_settings
        @info "Dataset: $(dataset), positive class label: $(posclass)"

        labelmap = (y) -> y == posclass
        train, test = load(dataset; labelmap = labelmap)
        (x_train, y_train) = reshape_for_python(train...);
        (x_test, y_test) = reshape_for_python(test...);

        for train_settings in dict_list_simple(Train_Settings)
            @unpack batchsize, epochs, seed, optimiser, steplength = train_settings
            @info "Batchsize: $(batchsize), runid: $(seed)"

            for model_settings in dict_list_simple(Model_Settings)
                @unpack type, arg, surrogate, reg, buffer = model_settings
                model_settings[:seed] = seed

                if optimiser <: Descent
                    optm = py"tf.keras.optimizers.SGD"
                else
                    @error "unknown optimiser"
                end

                # create model
                settings = Dict(
                    :fpr => arg,
                    :seed => seed,
                    :optimiser => optm,
                    :steplength => steplength,
                    :batchsize => batchsize,
                    :epochs => epochs
                )

                build_net = build_network_tfco(dataset)
                res = py"train_model"(build_net, x_train, y_train, x_test, y_test, settings, model_dir)
                s_train, s_test, elapsed_time = res

                save_simulation_tfco(
                    dataset_settings,
                    train_settings,
                    model_settings,
                    res,
                    y_train,
                    y_test,
                )
            end
        end
    end
    rm(model_dir; recursive=true)
end
