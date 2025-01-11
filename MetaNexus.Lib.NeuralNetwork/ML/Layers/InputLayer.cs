using MetaNexus.Lib.NeuralNetwork.ML.Layers.Abstractions;

public class InputLayer : Layer
{
    /// <summary>
    /// Конструктор для входного слоя.
    /// </summary>
    /// <param name="inputSize">Количество нейронов во входном слое.</param>
    public InputLayer(int inputSize) : base(inputSize) // Для входного слоя не нужно передавать inputSize
    {
    }

    /// <summary>
    /// Метод для выполнения прямого прохода через входной слой.
    /// </summary>
    /// <param name="input">Входные данные.</param>
    /// <returns>Те же самые входные данные.</returns>
    public override float[] Forward(float[] input)
    {
        return input; // Входной слой просто передает входные данные
    }
}
