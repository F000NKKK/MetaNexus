using MetaNexus.Lib.NeuralNetwork.ML.Abstractions;
using MetaNexus.Lib.NeuralNetwork.ML.Layers.Abstractions;
using MetaNexus.Lib.NeuralNetwork.ML.Layers;
using MetaNexus.Lib.NeuralNetwork.ML.Models;
using Newtonsoft.Json;

namespace MetaNexus.Lib.NeuralNetwork.ML
{
    /// <summary>
    /// Класс, представляющий нейронную сеть, которая строится на основе конфигурации из JSON.
    /// </summary>
    public class NeuralNetwork : INetwork
    {
        public List<Layer> Layers { get; private set; }

        /// <summary>
        /// Конструктор, принимающий строку JSON для конфигурации сети.
        /// </summary>
        /// <param name="json">Строка JSON, описывающая строение нейросети.</param>
        public NeuralNetwork(string json)
        {
            var config = JsonConvert.DeserializeObject<NetworkConfig>(json);
            Layers = new List<Layer>();

            // Создание слоев на основе конфигурации
            foreach (var layerConfig in config.Layers)
            {
                Layer layer = null;

                switch (layerConfig.Type.ToLower())
                {
                    case "input":
                        layer = new InputLayer(layerConfig.Size);
                        break;
                    case "dense":
                        layer = new DenseLayer(layerConfig.Size, layerConfig.Activation);
                        break;
                    default:
                        throw new InvalidOperationException($"Неизвестный тип слоя: {layerConfig.Type}");
                }

                Layers.Add(layer);
            }
        }

        /// <summary>
        /// Метод для выполнения прогноза (прямого прохода) через нейросеть.
        /// </summary>
        /// <param name="input">Входные данные для сети.</param>
        /// <returns>Выходные данные после прохождения через все слои сети.</returns>
        public float[] Predict(float[] input)
        {
            float[] output = input;

            foreach (var layer in Layers)
            {
                output = layer.Forward(output);
            }

            return output;
        }
    }
}
