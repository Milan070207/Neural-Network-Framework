from backend import xp
from layer import Layer


class Dropout(Layer):

    def __init__(self, rate):

        self.keep_rate = 1 - rate

    def forward(
            self,
            inputs,
            training=True):

        self.inputs = inputs

        if not training:

            self.output = inputs.copy()
            return

        self.mask = (
            xp.random.binomial(
                1,
                self.keep_rate,
                size=inputs.shape
            )
            / self.keep_rate
        )

        self.output = inputs * self.mask

    def backward(self, dvalues):

        self.dinputs = (
            dvalues *
            self.mask
        )