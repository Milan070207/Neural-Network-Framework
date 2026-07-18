from model_2 import Model
from dense import Dense
from activations import ReLU
from losses import SoftmaxCategoricalCrossEntropy
from optimizers import Adam
from datasets import Datasets
from utilities import plot_decision_boundary


X_train, y_train = Datasets.spiral(1000, 5)


model = Model()

model.add(Dense(2,64))
model.add(ReLU())

model.add(Dense(64,32))
model.add(ReLU())

model.add(Dense(32,5)) 


model.set(loss=SoftmaxCategoricalCrossEntropy(), optimizer=Adam())

plot_decision_boundary(model, X_train, y_train)
model.train(
    X_train,
    y_train,
    epochs=1000,
    batch_size=256,
    print_every=100
    )

plot_decision_boundary(model, X_train, y_train)

#model.save("spiral5.pkl")


