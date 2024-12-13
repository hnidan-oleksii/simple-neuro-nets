from importlib import import_module

import numpy as np
import pandas as pd
import seaborn as sns

from functions.helper_functions import forward, backward, update_weights
from genetic import GeneticAlgorithm


class NeuralNetwork():
    inputs: int
    outputs: int
    hidden: list

    learning_rate: float
    epochs: int
    thresh: float
    random_seed: int

    loss: callable
    loss_der: callable
    activation: callable
    activation_der: callable

    weights: list
    biases: list
    train_loss: np.array

    def __init__(self,
                 inputs: int,
                 outputs: int,
                 hidden: list,
                 learning_rate: int = 0.005,
                 epochs: int = 5000,
                 thresh: float = 0.75,
                 random_seed: int = 42,
                 loss_name: str = 'mse',
                 activation_name: str = 'sigmoid'):
        self.inputs = inputs
        self.outputs = outputs
        self.hidden = hidden

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.thresh = thresh
        self.random_seed = random_seed

        loss_funcs = import_module("functions.loss." + loss_name)
        self.loss = loss_funcs.loss
        self.loss_der = loss_funcs.loss_der
        act_funcs = import_module("functions.activation." + activation_name)
        self.activation = act_funcs.activation
        self.activation_der = act_funcs.activation_der

        self.weights = []
        self.biases = []
        self.train_loss = []

    def train(self,
              X: pd.DataFrame,
              y: pd.DataFrame,
              X_valid: pd.DataFrame,
              y_valid: pd.DataFrame,
              classes: list,
              init_type: str):

        X_np = X.to_numpy()
        y_np = y.to_numpy()
        X_valid_np = X_valid.to_numpy()
        y_valid_np = y_valid.to_numpy()

        features_num = X.shape[1]
        layers = [features_num] + self.hidden + [len(classes)]

        random = np.random.default_rng(seed=self.random_seed)

        if init_type == "genetic":
            def fitness_func(weights_vector):
                weights, biases = self.vector_to_weights(np.array(weights_vector), layers)
                activations = forward(X_np,
                                      weights,
                                      biases,
                                      self.activation)

                loss = np.mean(self.loss(activations[-1], y_np))
                return loss

            total_weights = sum(layers[i] * layers[i + 1] + layers[i + 1]
                                for i in range(len(layers) - 1))
            ga = GeneticAlgorithm(
                population_size=20,
                variables_number=total_weights,
                min_value=-0.5,
                max_value=0.5,
                generations=100
            )
            best_weights_vector = ga.run(fitness_func)
            self.weights, self.biases = self.vector_to_weights(np.array(best_weights_vector), layers)

        else:
            for i in range(len(layers) - 1):
                weight_shape = (layers[i], layers[i + 1])
                bias_shape = (1, layers[i + 1])

                if init_type == "xavier":
                    w = random.normal(0,
                                      np.sqrt(1. / (layers[i])),
                                      weight_shape)
                elif init_type == "uniform":
                    w = random.uniform(-0.5, 0.5, weight_shape)
                elif init_type == "constant":
                    w = np.zeros(weight_shape)

                b = np.zeros(bias_shape)
                self.weights.append(w)
                self.biases.append(b)

        for epoch in range(self.epochs):
            activations = forward(X_np,
                                  self.weights,
                                  self.biases,
                                  self.activation)

            deltas = backward(activations=activations,
                              weights=self.weights,
                              y_true=y_np,
                              activation_der=self.activation_der,
                              loss_der=self.loss_der)

            update_weights(weights=self.weights,
                           biases=self.biases,
                           activations=activations,
                           deltas=deltas,
                           learning_rate=self.learning_rate)

            y_pred = forward(X=X_valid_np,
                             weights=self.weights,
                             biases=self.biases,
                             activation_func=self.activation)[-1]
            loss = np.mean(np.square(y_valid_np - y_pred))
            self.train_loss.append(loss)

            if np.all((y_pred >= self.thresh) == y_valid_np):
                print("*"*50)
                print(f"Converged after {epoch+1} epochs")
                break

    def write_loss_graph_into_file(self, file_name: str):
        loss_plot = sns.lineplot(x=np.arange(1, len(self.train_loss) + 1),
                                 y=self.train_loss,
                                 markers=True)
        fig = loss_plot.get_figure()
        fig.savefig(file_name)
        fig.clf()

    def predict(self, X: pd.DataFrame):
        return forward(X=X,
                       weights=self.weights,
                       biases=self.biases,
                       activation_func=self.activation)[-1]

    def vector_to_weights(self, vector, layers):
        weights = []
        biases = []
        start = 0

        for i in range(len(layers) - 1):
            weight_size = layers[i] * layers[i + 1]
            bias_size = layers[i + 1]

            weights.append(vector[start:start + weight_size]
                           .reshape(layers[i], layers[i + 1]))
            start += weight_size

            biases.append(vector[start:start + bias_size]
                          .reshape(1, layers[i + 1]))
            start += bias_size

        return weights, biases
