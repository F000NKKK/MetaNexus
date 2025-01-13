using MetaNexus.Lib.NeuralNetwork.ML.Layers.Abstractions;
using MetaNexus.Lib.NeuralNetwork.Tensors;

public class InputLayer : Layer
{
    /// <summary>
    /// Конструктор для входного слоя.
    /// </summary>
    /// <param name="inputSize">Количество нейронов во входном слое.</param>
    public InputLayer(int inputSize) : base(inputSize)
    {
    }

    /// <summary>
    /// Метод для выполнения прямого прохода через входной слой.
    /// </summary>
    /// <param name="input">Входные данные в виде тензора.</param>
    /// <returns>Те же самые входные данные в виде тензора.</returns>
    public override Tensor Forward(Tensor input)
    {
        return input; // Входной слой просто передает входные данные
    }
}
