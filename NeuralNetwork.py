from DenseLayer import DenseLayer
from ActivationFunctions import ReLU, Softmax
import numpy as np

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

    def forward_propagate(self, x):
        for layer in self.layers:
            x = layer.forward(x)

        return x


def main():
    nn = NeuralNetwork([
        DenseLayer(4, 8, ReLU()),
        DenseLayer(8, 3, Softmax())
    ])

    x = np.random.randn(4, 5)  # 5 examples, 4 features each
    output = nn.forward_propagate(x)
    print(output)

if __name__ == "__main__":
    main()
