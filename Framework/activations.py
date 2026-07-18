from backend import xp
from layer import Layer

# ReLU activation
class ReLU(Layer):
    # Forward pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs
        
        # Calculate output values from inputs
        self.output = xp.maximum(0, inputs)
    
    # Backward pass
    def backward(self, dvalues):
        # Since we need to modify original variable,
        # let's make a copy of values first
        self.dinputs = dvalues.copy()
        
        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0

# Softmax activation
class Softmax(Layer):
    # Forward pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs
        # Get unnormalized probabilities
        exp_values = xp.exp(inputs - xp.max(inputs, axis=1, keepdims=True))
        # Normalize them for each sample
        probabilities = exp_values / xp.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
    # Backward pass
    def backward(self, dvalues):
        # Create uninitialized array
        self.dinputs = xp.empty_like(dvalues)
        
        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            
            # Calculate Jacobian matrix of the output and
            jacobian_matrix = xp.diagflat(single_output) - xp.dot(single_output, single_output.T)
            
            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            self.dinputs[index] = xp.dot(jacobian_matrix, single_dvalues)
            
class LeakyReLU(Layer):

    def __init__(self, alpha=0.01):

        self.alpha = alpha

    def forward(self, inputs, training=True):

        self.inputs = inputs

        self.output = xp.where(
            inputs > 0,
            inputs,
            self.alpha * inputs
        )

    def backward(self, dvalues):

        self.dinputs = dvalues.copy()

        self.dinputs[self.inputs < 0] *= self.alpha
        
class ELU(Layer):

    def __init__(self, alpha=1.0):

        self.alpha = alpha

    def forward(self, inputs, training=True):

        self.inputs = inputs

        self.output = xp.where(
            inputs > 0,
            inputs,
            self.alpha * (xp.exp(inputs)-1)
        )

    def backward(self, dvalues):

        self.dinputs = dvalues.copy()

        self.dinputs *= xp.where(
            self.inputs > 0,
            1,
            self.output + self.alpha
        )
        
        
class GELU(Layer):

    def forward(self, inputs, training=True):

        self.inputs = inputs

        self.output = (
            0.5 *
            inputs *
            (
                1 +
                xp.tanh(
                    xp.sqrt(2/xp.pi) *
                    (
                        inputs +
                        0.044715 *
                        inputs**3
                    )
                )
            )
        )

class Sigmoid(Layer):

    def forward(self, inputs, training=True):

        self.output = 1 / (
            1 + xp.exp(-inputs)
        )

    def backward(self, dvalues):

        self.dinputs = (
            dvalues *
            self.output *
            (1-self.output)
        )

class Tanh(Layer):

    def forward(self, inputs, training=True):

        self.output = xp.tanh(inputs)

    def backward(self, dvalues):

        self.dinputs = (
            dvalues *
            (1-self.output**2)
        )
        
class Softplus(Layer):

    def forward(self, inputs, training=True):

        self.inputs = inputs

        self.output = xp.log(
            1 + xp.exp(inputs)
        )

    def backward(self, dvalues):

        sigmoid = 1 / (
            1 + xp.exp(-self.inputs)
        )

        self.dinputs = (
            dvalues *
            sigmoid
        )
        
class Swish(Layer):

    def forward(self, inputs, training=True):

        self.inputs = inputs

        self.sigmoid = (
            1 /
            (1+xp.exp(-inputs))
        )

        self.output = (
            inputs *
            self.sigmoid
        )

    def backward(self, dvalues):

        self.dinputs = (
            dvalues *
            (
                self.sigmoid +
                self.output *
                (
                    1 -
                    self.sigmoid
                )
            )
        )