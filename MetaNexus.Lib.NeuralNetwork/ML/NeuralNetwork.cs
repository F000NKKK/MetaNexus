using MetaNexus.Lib.NeuralNetwork.ML.Abstractions;
using MetaNexus.Lib.NeuralNetwork.ML.Layers.Abstractions;
using MetaNexus.Lib.NeuralNetwork.Tensors;

public class NeuralNetwork : INetwork
{
    private List<ILayer> layers;

    public NeuralNetwork()
    {
        layers = new List<ILayer>();
    }

    /// <summary>
    /// Метод для выполнения прогноза через сеть.
    /// </summary>
    /// <param name="input">Входной тензор.</param>
    /// <returns>Выходной тензор.</returns>
    public Tensor Predict(Tensor input)
    {
        Tensor output = input;

        // Проходим по всем слоям
        foreach (var layer in layers)
        {
            output = layer.Forward(output);
        }

        return output;
    }

    /// <summary>
    /// Метод для обучения сети на основе входных и целевых данных.
    /// </summary>
    /// <param name="input">Входной тензор для обучения.</param>
    /// <param name="target">Целевой тензор для обучения.</param>
    /// <param name="learningRate">Коэффициент обучения для обновления весов.</param>
    public void Train(Tensor input, Tensor target, float learningRate)
    {
        // Прямой проход
        Tensor output = Predict(input);

        // Вычисление ошибки на выходе
        Tensor error = target - output;

        // Обратное распространение
        Tensor delta = error * output.Derivative(); // Вычисление delta (градиент)

        // Пройдемся по слоям в обратном порядке
        for (int i = layers.Count - 1; i >= 0; i--)
        {
            var layer = layers[i];
            delta = layer.Backward(delta, learningRate);
        }
    }

    /// <summary>
    /// Метод для добавления слоя в нейронную сеть.
    /// </summary>
    /// <param name="layer">Слой для добавления в сеть.</param>
    public void AddLayer(ILayer layer)
    {
        layers.Add(layer);
    }
}
