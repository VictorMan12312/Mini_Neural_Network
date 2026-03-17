class BaseOptimizer:
    """Abstract optimizer class."""
    def step(self, layers):
        raise NotImplementedError