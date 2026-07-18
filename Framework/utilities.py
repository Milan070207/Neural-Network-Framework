from backend import xp
import matplotlib.pyplot as plt

def plot_dataset(X, y):

    plt.figure(figsize=(6,6))

    plt.scatter(
        X[:,0],
        X[:,1],
        c=y,
        cmap="brg",
        edgecolors="black"
    )

    plt.show()
    
def plot_decision_boundary(
        model,
        X,
        y,
        resolution=300):
    x_min = X[:,0].min() - 1
    x_max = X[:,0].max() + 1

    y_min = X[:,1].min() - 1
    y_max = X[:,1].max() + 1
    
    xx, yy = xp.meshgrid(
    xp.linspace(
        x_min,
        x_max,
        resolution
    ),
    xp.linspace(
        y_min,
        y_max,
        resolution
    )
    )
    
    grid = xp.c_[
    xx.ravel(),
    yy.ravel()]
    
    predictions = model.predict(grid)
    
    predictions = xp.argmax(
    predictions,
    axis=1)
    
    predictions = predictions.reshape(
    xx.shape)
    plt.figure(figsize=(8,8))

    plt.contourf(
        xx,
        yy,
        predictions,
        alpha=0.35,
        cmap="brg"
    )

    plt.scatter(
        X[:,0],
        X[:,1],
        c=y,
        edgecolors="black",
        cmap="brg"
    )

    plt.show()
    
def plot_history(history):
    plt.figure(figsize=(10,5))

    plt.subplot(121)

    plt.plot(history)

    plt.legend([
        "Training",
        "Validation"
    ])

    plt.xlabel("Epoch")

    plt.ylabel("Loss")
    
def plot_weights(model):
    for layer in model.layers:

        if not layer.trainable:
            continue

        plt.hist(
            layer.weights.flatten(),
            bins=50
        )

        plt.show()
    
def plot_gradients(model):
    for layer in model.layers:

        if not layer.trainable:
            continue

        plt.hist(
            layer.dweights.flatten(),
            bins=50
        )

        plt.show()