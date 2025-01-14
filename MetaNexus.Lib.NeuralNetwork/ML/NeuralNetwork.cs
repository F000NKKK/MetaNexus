using MetaNexus.Lib.NeuralNetwork.ML.Abstractions;
using MetaNexus.Lib.NeuralNetwork.ML.Layers.Abstractions;
using MetaNexus.Lib.NeuralNetwork.ML.Models;
using MetaNexus.Lib.NeuralNetwork.Tensors;
using Newtonsoft.Json;

public class NeuralNetwork : INetwork
{
    private List<ILayer> layers;

    public NeuralNetwork()
    {
        layers = new List<ILayer>();
    }

    public NeuralNetwork(string jsonConfig)
    {
        layers = new List<ILayer>();

        // Десериализация JSON-конфигурации
        var config = JsonConvert.DeserializeObject<NetworkConfig>(jsonConfig);

        if (config == null)
        {
            throw new ArgumentException("Ошибка в конфигурации нейронной сети.");
        }

        // Создание слоев на основе конфигурации
        int inputSize = config.InputSize;

        foreach (var layerConfig in config.Layers)
        {
            ILayer layer = null;

            // Используем словарь для сопоставления строк с функциями
            var activationFunctions = new Dictionary<string, Layer.ActivationFunc>
            {
                { "relu", Tensor.ApplyReLUStatic },
                { "sigmoid", Tensor.ApplySigmoidStatic },
                { "tanh", Tensor.ApplyTanhStatic },
                { "softmax", Tensor.ApplySoftmaxStatic },
                { "swish", Tensor.ApplySwishStatic },
                { "identity", Tensor.ApplyIdentityStatic }
            };

            var activationPrimeFunctions = new Dictionary<string, Layer.ActivationPrimeFunc>
            {
                { "relu", Tensor.ApplyReLUPrimeStatic },
                { "sigmoid", Tensor.ApplySigmoidPrimeStatic },
                { "tanh", Tensor.ApplyTanhPrimeStatic },
                { "softmax", Tensor.ApplySoftplusPrimeStatic }, // Пример для Softmax
                { "swish", Tensor.ApplySwishPrimeStatic },
                { "identity", Tensor.ApplyIdentityStatic }
            };

            if (!activationFunctions.ContainsKey(layerConfig.Activation) ||
                !activationPrimeFunctions.ContainsKey(layerConfig.Activation))
            {
                throw new ArgumentException($"Неизвестная функция активации или её производная: {layerConfig.Activation}");
            }

            Layer.ActivationFunc activationFunc = activationFunctions[layerConfig.Activation];
            Layer.ActivationPrimeFunc activationPrimeFunc = activationPrimeFunctions[layerConfig.Activation];

            if (layerConfig.Type == "input")
            {
                layer = new InputLayer(inputSize, layerConfig.Size, activationFunc, activationPrimeFunc);
                inputSize = layerConfig.Size;
            }
            else if (layerConfig.Type == "dense")
            {
                layer = new DenseLayer(inputSize, layerConfig.Size, activationFunc, activationPrimeFunc);
                inputSize = layerConfig.Size;
            }
            else
            {
                throw new ArgumentException($"Неизвестный тип слоя: {layerConfig.Type}");
            }

            layers.Add(layer);
        }
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

        // Вычисление delta для последнего слоя
        var lastLayer = layers[^1];
        Tensor delta = error * lastLayer.ApplyActivationPrime(output);

        // Обратное распространение
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
