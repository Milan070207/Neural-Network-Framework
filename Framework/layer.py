from backend import xp

class Layer:
    """
    Base class for every layer.
    """

    trainable = False

    def forward(self, inputs, training=True):
        self.inputs = inputs

        if self.gamma is None:
            self.build(inputs.shape[1])

        if training:

            self.batch_mean = xp.mean(
                inputs,
                axis=0,
                keepdims=True
            )

            self.batch_var = xp.var(
                inputs,
                axis=0,
                keepdims=True
            )

            self.running_mean = (
                self.momentum *
                self.running_mean
                +
                (1 - self.momentum) *
                self.batch_mean
            )

            self.running_var = (
                self.momentum *
                self.running_var
                +
                (1 - self.momentum) *
                self.batch_var
            )

            mean = self.batch_mean
            var = self.batch_var

        else:

            mean = self.running_mean
            var = self.running_var

        self.x_hat = (
            inputs - mean
        ) / xp.sqrt(
            var + self.epsilon
        )

        self.output = (
            self.gamma *
            self.x_hat
            +
            self.beta
        )

    def backward(self, dvalues):
        N = dvalues.shape[0]

        self.dgamma = xp.sum(
            dvalues * self.x_hat,
            axis=0,
            keepdims=True
        )

        self.dbeta = xp.sum(
            dvalues,
            axis=0,
            keepdims=True
        )

        dxhat = dvalues * self.gamma

        inv_std = 1 / xp.sqrt(
            self.batch_var + self.epsilon
        )

        dvar = xp.sum(
            dxhat *
            (self.inputs - self.batch_mean) *
            -0.5 *
            inv_std**3,
            axis=0,
            keepdims=True
        )

        dmean = (
            xp.sum(
                -dxhat * inv_std,
                axis=0,
                keepdims=True
            )
            +
            dvar *
            xp.mean(
                -2 *
                (self.inputs - self.batch_mean),
                axis=0,
                keepdims=True
            )
        )

        self.dinputs = (
            dxhat * inv_std
            +
            dvar *
            2 *
            (self.inputs - self.batch_mean)
            / N
            +
            dmean / N
        )

    def regularization_loss(self):
        return 0.0


class TrainableLayer(Layer):

    trainable = True

    def __init__(
            self,
            weight_regularizer_l2=0,
            bias_regularizer_l2=0):

        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l2 = bias_regularizer_l2

    def regularization_loss(self):

        loss = 0

        if self.weight_regularizer_l2 > 0:
            loss += (
                self.weight_regularizer_l2 *
                xp.sum(self.weights ** 2)
            )

        if self.bias_regularizer_l2 > 0:
            loss += (
                self.bias_regularizer_l2 *
                xp.sum(self.biases ** 2)
            )

        return loss

