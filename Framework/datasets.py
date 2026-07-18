from backend import xp


class Datasets:
    @staticmethod
    def mnist(path):

        data = xp.load(path)

        X_train = data["x_train"]
        y_train = data["y_train"]

        X_test = data["x_test"]
        y_test = data["y_test"]

        # flatten images
        X_train = X_train.reshape(-1,784)
        X_test = X_test.reshape(-1,784)

        # normalize
        X_train = X_train.astype(xp.float32)/255.0
        X_test = X_test.astype(xp.float32)/255.0

        return (
            xp.asarray(X_train),
            xp.asarray(y_train),
            xp.asarray(X_test),
            xp.asarray(y_test)
        )

    @staticmethod
    def spiral(samples, classes):

        X = xp.zeros((samples * classes, 2))
        y = xp.zeros(samples * classes, dtype=xp.uint8)

        for class_number in range(classes):

            ix = range(
                samples * class_number,
                samples * (class_number + 1)
            )

            r = xp.linspace(0, 1, samples)

            t = xp.linspace(
                class_number * 4,
                (class_number + 1) * 4,
                samples
            )

            t += xp.random.randn(samples) * 0.2

            X[ix] = xp.c_[
                r * xp.sin(t * 2.5),
                r * xp.cos(t * 2.5)
            ]

            y[ix] = class_number

        return X, y
    
    @staticmethod
    def vertical(samples, classes):

        X = xp.zeros((samples * classes, 2))
        y = xp.zeros(samples * classes, dtype=xp.uint8)

        for class_number in range(classes):

            ix = range(
                samples * class_number,
                samples * (class_number + 1)
            )

            X[ix] = xp.c_[
                xp.random.randn(samples) * 0.1
                + class_number / 3,

                xp.random.randn(samples)
            ]

            y[ix] = class_number

        return X, y
    
    @staticmethod
    def xor(samples=1000):

        X = xp.random.uniform(
            -1,
            1,
            (samples, 2)
        )

        y = (
            (X[:,0] > 0) ^
            (X[:,1] > 0)
        ).astype(xp.uint8)

        return X, y
    
    @staticmethod
    def train_test_split(
            X,
            y,
            test_size=0.2,
            shuffle=True):

        if shuffle:

            indices = xp.random.permutation(len(X))

            X = X[indices]
            y = y[indices]

        split = int(len(X) * (1 - test_size))

        return (
            X[:split],
            X[split:],
            y[:split],
            y[split:]
        )