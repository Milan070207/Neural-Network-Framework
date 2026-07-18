from backend import xp

import pickle


class Model:

    def __init__(self):

        self.layers = []
        self.loss = None
        self.optimizer = None
        self.accuracy = None
    
    def get_parameters(self):
        """Collect (weights, biases) for every layer that has them, as numpy arrays."""
        parameters = []
        for layer in self.layers:
            if hasattr(layer, "weights"):
                weights = layer.weights
                biases = layer.biases
                # Convert cupy -> numpy so the saved file doesn't require a GPU to load
                if hasattr(weights, "get"):
                    weights = weights.get()
                if hasattr(biases, "get"):
                    biases = biases.get()
                parameters.append((weights, biases))
        return parameters

    def set_parameters(self, parameters):
        """Load (weights, biases) pairs back into each trainable layer, in order."""
        trainable_layers = [layer for layer in self.layers if hasattr(layer, "weights")]

        if len(parameters) != len(trainable_layers):
            raise ValueError(
                f"Parameter count mismatch: file has {len(parameters)} layers, "
                f"model has {len(trainable_layers)} trainable layers."
            )

        for (weights, biases), layer in zip(parameters, trainable_layers):
            layer.weights = xp.array(weights)
            layer.biases = xp.array(biases)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.get_parameters(), f)

    def load(self, path):
        with open(path, "rb") as f:
            parameters = pickle.load(f)
        self.set_parameters(parameters)

    def add(self, layer):

        self.layers.append(layer)

    def set(self, *, loss=None, optimizer=None, accuracy=None):

        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

    def forward(self, X, training=True):

        output = X

        for layer in self.layers:
            layer.forward(output)
            output = layer.output

        return output

    def backward(self, output, y):
        self.loss.backward(self.loss.output, y)

        gradient = self.loss.dinputs
    
        for layer in reversed(self.layers):
            layer.backward(gradient)
            gradient = layer.dinputs

    def train(
        self,
        X,
        y,
        *,
        epochs=1000,
        batch_size=None,
        print_every=100,
        early_stopping_patience=100,
        min_delta=0.0):

        if batch_size is None:
            batch_size = len(X)

        steps = len(X) // batch_size

        if steps * batch_size < len(X):
            steps += 1

        best_loss = float("inf")
        best_parameters = None
        epochs_without_improvement = 0

        for epoch in range(epochs+1):

            epoch_loss = 0
            epoch_acc = 0

            for step in range(steps):

                batch_X = X[
                    step * batch_size:
                    (step + 1) * batch_size
                ]

                batch_y = y[
                    step * batch_size:
                    (step + 1) * batch_size
                ]

                output = self.forward(batch_X)

                data_loss = self.loss.forward(output, batch_y)

                reg_loss = 0

                for layer in self.layers:

                    if hasattr(layer, "regularization_loss"):
                        reg_loss += layer.regularization_loss()

                loss = data_loss + reg_loss

                predictions = xp.argmax(output, axis=1)

                if len(batch_y.shape) == 2:
                    labels = xp.argmax(batch_y, axis=1)
                else:
                    labels = batch_y

                accuracy = xp.mean(predictions == labels)

                self.backward(output, batch_y)

                self.optimizer.pre_update_params()

                for layer in self.layers:

                    if hasattr(layer, "weights"):
                        self.optimizer.update_params(layer)

                self.optimizer.post_update_params()

                epoch_loss += loss
                epoch_acc += accuracy

            epoch_loss /= steps
            epoch_acc /= steps

            # Pull to a plain python float so comparisons/formatting work
            # the same whether xp is numpy or cupy
            epoch_loss_value = float(epoch_loss)

            if epoch % print_every == 0:
                
                print(
                    f"epoch {epoch:5d} | "
                    f"loss {epoch_loss:.4f} | "
                    f"acc {epoch_acc:.3f} | "
                    f"lr {self.optimizer.current_learning_rate:.6f}"
                )

            # --- early stopping bookkeeping ---
            if epoch_loss_value < best_loss - min_delta:
                best_loss = epoch_loss_value
                best_parameters = self.get_parameters()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if best_loss == 0 or epochs_without_improvement >= early_stopping_patience:
                print(
                    f"Early stopping at epoch {epoch} | "
                    f"best loss {best_loss:.6f}"
                )
                break

        # Always finish on the best weights ever seen, not whatever
        # the last epoch happened to land on
        if best_parameters is not None:
            self.set_parameters(best_parameters)
                
    def build(self, n_features):

        self.gamma = xp.ones((1, n_features))
        self.beta = xp.zeros((1, n_features))

        self.running_mean = xp.zeros((1, n_features))
        self.running_var = xp.ones((1, n_features))
        
    def predict(self, X):

        return self.forward(
            X,
            training=False
        )