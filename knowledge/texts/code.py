class Counter:
    def __init__(self):
        self.total = 0

    def add(self, x):
        self.total = self.total + x
        return self.total

def mean(values):
    if not values:
        return 0.0
    return sum(values) / len(values)
