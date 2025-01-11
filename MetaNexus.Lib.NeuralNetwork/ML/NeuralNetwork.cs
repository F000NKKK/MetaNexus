using MetaNexus.Lib.NeuralNetwork.ML.Abstractions;
using MetaNexus.Lib.NeuralNetwork.ML.Layers.Abstractions;
using MetaNexus.Lib.NeuralNetwork.ML.Layers;
using MetaNexus.Lib.NeuralNetwork.ML.Models;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;

namespace MetaNexus.Lib.NeuralNetwork.ML
{
    public class NeuralNetwork : INetwork
    {
        public List<Layer> Layers { get; private set; }

        /// <summary>
        /// Конструктор для создания нейронной сети на основе JSON конфигурации.
        /// </summary>
        /// <param name="json">Строка JSON, описывающая строение нейросети.</param>
        public NeuralNetwork(string json)
        {
            var config = JsonConvert.DeserializeObject<NetworkConfig>(json);
            Layers = new List<Layer>();

            int inputSize = config.InputSize;  // Размер входных данных

            foreach (var layerConfig in config.Layers)
            {
                Layer layer = CreateLayerFromConfig(layerConfig, inputSize);
                Layers.Add(layer);
                inputSize = layer.Size;  // Устанавливаем новый inputSize для следующего слоя
            }
        }

        /// <summary>
        /// Конструктор для создания нейронной сети из бинарного файла, содержащего веса и смещения.
        /// </summary>
        /// <param name="json">Строка JSON, описывающая строение нейросети.</param>
        /// <param name="binaryFilePath">Путь к бинарному файлу с весами и смещениями.</param>
        /// <param name="loadWeights">Флаг для указания загрузки весов из бинарного файла.</param>
        public NeuralNetwork(string json, string binaryFilePath, bool loadWeights)
        {
            var config = JsonConvert.DeserializeObject<NetworkConfig>(json);
            Layers = new List<Layer>();

            int inputSize = config.InputSize;  // Размер входных данных

            foreach (var layerConfig in config.Layers)
            {
                Layer layer = CreateLayerFromConfig(layerConfig, inputSize);
                Layers.Add(layer);
                inputSize = layer.Size;  // Устанавливаем новый inputSize для следующего слоя
            }

            if (loadWeights)
            {
                LoadWeightsFromFile(binaryFilePath);
            }
            else
            {
                throw new InvalidOperationException("Для загрузки весов необходимо передать путь к бинарному файлу.");
            }
        }

        private Layer CreateLayerFromConfig(LayerConfig layerConfig, int inputSize)
        {
            switch (layerConfig.Type.ToLower())
            {
                case "input":
                    return new InputLayer(inputSize);  // Используем inputSize для первого слоя
                case "dense":
                    return new DenseLayer(inputSize, layerConfig.Size, layerConfig.Activation);  // Передаем inputSize для Dense
                default:
                    throw new InvalidOperationException($"Неизвестный тип слоя: {layerConfig.Type}");
            }
        }

        private void LoadWeightsFromFile(string binaryFilePath)
        {
            if (!File.Exists(binaryFilePath))
            {
                throw new FileNotFoundException($"Не удалось найти бинарный файл: {binaryFilePath}");
            }

            using (FileStream fs = new FileStream(binaryFilePath, FileMode.Open))
            {
                var reader = new BinaryReader(fs);

                foreach (var layer in Layers)
                {
                    layer.LoadWeights(reader);
                }
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
                Console.WriteLine($"Выход после слоя {layer.GetType().Name}: {string.Join(", ", output)}");
            }

            return output;
        }

        /// <summary>
        /// Сохранение весов нейросети в бинарный файл.
        /// </summary>
        /// <param name="binaryFilePath">Путь к файлу, в который будут сохранены веса и конфигурация.</param>
        public void SaveWeights(string binaryFilePath)
        {
            using (FileStream fs = new FileStream(binaryFilePath, FileMode.Create))
            {
                var writer = new BinaryWriter(fs);

                // Сохраняем количество слоев
                writer.Write(Layers.Count);

                foreach (var layer in Layers)
                {
                    // Сохраняем информацию о слое: размер и тип
                    writer.Write(layer.Size);

                    writer.Write(layer.GetType().Name.ToLower());

                    if (layer is DenseLayer denseLayer)
                    {
                        // Сохраняем функцию активации для слоя Dense
                        writer.Write(denseLayer.Activation);
                    }

                    // Сохраняем веса слоя
                    layer.SaveWeights(writer);
                }
            }
        }
    }
}
