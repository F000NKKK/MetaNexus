using MetaNexus.Lib.NeuralNetwork.Math.Normalizers;
using MetaNexus.Lib.NeuralNetwork.Math.Normalizers.Abstractions;
using MetaNexus.Lib.NeuralNetwork.ML;

namespace MetaNexus.App.DemoApp
{
    internal class Program
    {
        static void Main(string[] args)
        {
            // Пример конфигурации нейронной сети в формате JSON
            string jsonConfig = @"
            {
                'InputSize': 3, 
                'Layers': [
                    { 'Type': 'input', 'Size': 2 },
                    { 'Type': 'dense', 'Size': 3, 'Activation': 'relu' },
                    { 'Type': 'dense', 'Size': 1, 'Activation': 'sigmoid' }
                ]
            }";

            // Путь к бинарному файлу с весами и смещениями
            string binaryWeightsPath = "weights.bin";

            try
            {
                // Проверка существования файла весов
                if (!File.Exists(binaryWeightsPath))
                {
                    Console.WriteLine($"Бинарный файл {binaryWeightsPath} не найден, создаем новый...");

                    // Создание нейронной сети и сохранение весов в файл
                    var neuralNetwork = new NeuralNetwork(jsonConfig);
                    neuralNetwork.SaveWeights(binaryWeightsPath); // Сохраняем веса в бинарный файл
                    Console.WriteLine($"Файл с весами {binaryWeightsPath} успешно создан.");
                }

                // Теперь загружаем веса из файла
                var networkWithWeights = new NeuralNetwork(jsonConfig, binaryWeightsPath, true);

                float[] input = { 1.0f, 0.5f, -2.0f };

                // Пример нормализации входных данных
                var minMaxNormalizer = new MinMaxNormalizer(input);
                float[] normalizedInputData = minMaxNormalizer.Normalize(input);

                // Выполнение прогноза
                var output = networkWithWeights.Predict(normalizedInputData);

                // Пример денормализации выходных данных
                float[] denormalizedOutputData = minMaxNormalizer.Denormalize(output);

                // Вывод результатов
                Console.WriteLine("Прогноз сети:");
                foreach (var value in denormalizedOutputData)
                {
                    Console.WriteLine(value);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Ошибка: {ex.Message}");
            }
        }
    }
}
