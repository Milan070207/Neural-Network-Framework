from backend import xp
from activations import Softmax

# Common loss class
class Loss:
    def regularization_loss(self, layer):

        regularization_loss = 0

        if layer.l2 > 0:
            regularization_loss += (
                layer.l2 *
                xp.sum(layer.weights ** 2)
            )

        return regularization_loss
    # Calculates the data and regularization losses
    # given model output and ground truth values
    def calculate(self, output, y):
        # Calculate sample losses
        
        sample_losses = self.forward(output, y)
        # Calculate mean loss
        data_loss = xp.mean(sample_losses)
        # Return loss
        return data_loss
    
# Cross-entropy loss
class CategoricalCrossEntropy(Loss):
    # Forward pass
    def forward(self, y_pred, y_true):
        # Number of samples in a batch
        samples = len(y_pred)

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = xp.clip(y_pred, 1e-7, 1 - 1e-7)
        
        # Probabilities for target values -
        # only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
            
        # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = xp.sum(y_pred_clipped * y_true, axis=1)
            
        # Losses
        negative_log_likelihoods = -xp.log(correct_confidences)
        return negative_log_likelihoods
    
    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        
        # Number of labels in every sample
        # We'll use the first sample to count them
        labels = len(dvalues[0])
        
        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = xp.eye(labels)[y_true]
            
        # Calculate gradient
        self.dinputs = -y_true / dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples

# Softmax classifier - combined Softmax activation
# and cross-entropy loss for faster backward step
class SoftmaxCategoricalCrossEntropy():
    # Creates activation and loss function objects
    def __init__(self):
        self.activation = Softmax()
        self.loss = CategoricalCrossEntropy()
        
    # Forward pass
    def forward(self, ixputs, y_true):
        # Output layer's activation function
        self.activation.forward(ixputs)
    
        # Set the output
        self.output = self.activation.output
    
        # Calculate and return loss value
        return self.loss.calculate(self.output, y_true)
    
    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        
        # If labels are one-hot encoded,
        # turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = xp.argmax(y_true, axis=1)
            
        # Copy so we can safely modify
        self.dinputs = dvalues.copy()
        
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        
        # Normalize gradient
        self.dinputs = self.dinputs / samples