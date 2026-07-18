from backend import xp
from layer import TrainableLayer


class BatchNormalization(TrainableLayer):
    def build(self, n_features):

        self.gamma = xp.ones((1, n_features))
        self.beta = xp.zeros((1, n_features))

        self.running_mean = xp.zeros((1, n_features))
        self.running_var = xp.ones((1, n_features))
    def __init__(
            self,
            momentum=0.9,
            epsilon=1e-5):

        super().__init__()

        self.momentum = momentum
        self.epsilon = epsilon

        self.gamma = None
        self.beta = None

        self.running_mean = None
        self.running_var = None