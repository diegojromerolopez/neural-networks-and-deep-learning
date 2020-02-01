from network import Network
import mnist_loader


def train(hidden_layers):
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    network = Network([784] + hidden_layers)
    network.SGD(training_data, epochs=30, learning_rate=10, mini_batch_size=3.0, test_data=test_data)


if __name__ == "__main__":
    train([30, 10])
