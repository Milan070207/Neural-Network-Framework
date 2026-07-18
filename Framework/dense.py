from backend import xp
from layer import TrainableLayer

class Dense(TrainableLayer):

    def __init__(
            self,
            n_inputs,
            n_neurons,
            initializer="he",
            weight_regularizer_l2=0,
            bias_regularizer_l2=0):

        super().__init__(
            weight_regularizer_l2,
            bias_regularizer_l2
        )

        if initializer == "he":
            scale = xp.sqrt(2 / n_inputs)

        elif initializer == "xavier":
            scale = xp.sqrt(1 / n_inputs)

        else:
            scale = 0.01

        self.weights = (
            xp.random.randn(
                n_inputs,
                n_neurons
            ) * scale
        )

        self.biases = xp.zeros((1, n_neurons))

    def forward(self, inputs, training=True):

        self.inputs = inputs
        self.output = (
            xp.dot(inputs, self.weights)
            + self.biases
        )

    def backward(self, dvalues):

        self.dweights = xp.dot(
            self.inputs.T,
            dvalues
        )

        self.dbiases = xp.sum(
            dvalues,
            axis=0,
            keepdims=True
        )

        if self.weight_regularizer_l2 > 0:
            self.dweights += (
                2 *
                self.weight_regularizer_l2 *
                self.weights
            )

        if self.bias_regularizer_l2 > 0:
            self.dbiases += (
                2 *
                self.bias_regularizer_l2 *
                self.biases
            )

        self.dinputs = xp.dot(
            dvalues,
            self.weights.T
        )