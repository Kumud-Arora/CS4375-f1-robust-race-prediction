import numpy as np

class Node:
    def __init__(self):
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None


class RegressionTree:
    def __init__(self, max_depth=3, min_samples_split=10):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def mse(self, y):
        if len(y) == 0:
            return 0
        return np.var(y) * len(y)

    def best_split(self, X, y):
        n, p = X.shape
        best_gain = -1e9
        best_feature = None
        best_thresh = None

        parent_error = self.mse(y)

        for f in range(p):
            values = np.unique(X[:, f])

            if len(values) > 30:
                values = np.percentile(values, np.linspace(0, 100, 30))

            for t in values:
                left = X[:, f] <= t
                right = ~left

                if np.sum(left) < self.min_samples_split:
                    continue
                if np.sum(right) < self.min_samples_split:
                    continue

                gain = parent_error - self.mse(y[left]) - self.mse(y[right])

                if gain > best_gain:
                    best_gain = gain
                    best_feature = f
                    best_thresh = t

        return best_feature, best_thresh

    def build(self, X, y, depth):
        node = Node()

        if depth >= self.max_depth or len(y) < self.min_samples_split:
            node.value = np.mean(y)
            return node

        feature, thresh = self.best_split(X, y)

        if feature is None:
            node.value = np.mean(y)
            return node

        left = X[:, feature] <= thresh
        right = ~left

        node.feature = feature
        node.threshold = thresh

        node.left = self.build(X[left], y[left], depth + 1)
        node.right = self.build(X[right], y[right], depth + 1)

        return node

    def fit(self, X, y):
        self.root = self.build(X, y, 0)
        return self

    def predict_one(self, x, node):
        if node.value is not None:
            return node.value

        if x[node.feature] <= node.threshold:
            return self.predict_one(x, node.left)
        return self.predict_one(x, node.right)

    def predict(self, X):
        return np.array([self.predict_one(x, self.root) for x in X])


class GradientBoostingClassifier:
    def __init__(self, n_estimators=50, learning_rate=0.1, max_depth=3, subsample=1.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample

        self.trees = []
        self.base = 0

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def log_odds(self, y):
        p = np.mean(y)
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        self.base = self.log_odds(y)
        pred = np.full(len(y), self.base)

        self.trees = []

        for _ in range(self.n_estimators):
            residuals = y - self.sigmoid(pred)

            if self.subsample < 1:
                index = np.random.choice(len(y), int(len(y) * self.subsample), replace=False)
                Xb = X[index]
                rb = residuals[index]
            else:
                Xb = X
                rb = residuals

            tree = RegressionTree(self.max_depth)
            tree.fit(Xb, rb)

            update = tree.predict(X)
            pred += self.learning_rate * update

            self.trees.append(tree)

        return self

    def predict_proba(self, X):
        X = np.array(X)

        pred = np.full(len(X), self.base)
        for t in self.trees:
            pred += self.learning_rate * t.predict(X)

        p1 = self.sigmoid(pred)
        return np.c_[1 - p1, p1]

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] > threshold).astype(int)