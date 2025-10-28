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
        DenseLayer(input_size=8, output_size=4, activation_function=ReLU()),
        DenseLayer(input_size=4, output_size=2, activation_function=Softmax())
    ])
    
    print(nn.layers[0].weights.shape)
    print(nn.layers[1].weights.shape)

    x = np.random.randn(8, 5)  # 5 examples, 8 features each
    print(x.shape)
    output = nn.forward_propagate(x)
    print(output)

if __name__ == "__main__":
    main()
