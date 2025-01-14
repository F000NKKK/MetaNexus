using MetaNexus.Lib.NeuralNetwork.ML.Layers.Abstractions;
using MetaNexus.Lib.NeuralNetwork.Tensors;

public class InputLayer : Layer
{
    public InputLayer(int inputSize, int size, ActivationFunc activationFunction, ActivationPrimeFunc activationPrimeFunction) : base(inputSize, size, activationFunction, activationPrimeFunction)
    {
    }

    /// <summary>
    /// Метод для выполнения прямого прохода через входной слой.
    /// </summary>
    /// <param name="input">Входные данные в виде тензора.</param>
    /// <returns>Те же самые входные данные в виде тензора.</returns>
    public override Tensor Forward(Tensor input)
    {
        this.input = input;
        return input; // Входной слой просто передает входные данные
    }
}
