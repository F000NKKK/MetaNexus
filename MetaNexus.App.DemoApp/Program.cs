using MetaNexus.Lib.NeuralNetwork.Tensors;

namespace SimpleNeuralNetwork
{
    class Program
    {
        static NeuralNetwork nn = new NeuralNetwork();

        static void Main(string[] args)
        {
            // Архитектура сети с уменьшенными слоями, чтобы избежать переполнения
            nn.AddLayer(new InputLayer(2, 2, Tensor.ApplyIdentityStatic, Tensor.ApplyIdentityStatic)); // 2 нейрона во входном слое
            nn.AddLayer(new DenseLayer(2, 512, Tensor.ApplyIdentityStatic, Tensor.ApplyIdentityStatic)); // 512 нейронов во входном слое
            nn.AddLayer(new DenseLayer(512, 1024, Tensor.ApplySwishStatic, Tensor.ApplySwishPrimeStatic)); // 1024 нейрона в скрытом слое
            nn.AddLayer(new DenseLayer(1024, 2048, Tensor.ApplySwishStatic, Tensor.ApplySwishPrimeStatic)); // 2048 нейронов в скрытом слое
            nn.AddLayer(new DenseLayer(2048, 4096, Tensor.ApplySwishStatic, Tensor.ApplySwishPrimeStatic)); // 4096 нейронов в скрытом слое
            nn.AddLayer(new DenseLayer(4096, 8192, Tensor.ApplySwishStatic, Tensor.ApplySwishPrimeStatic)); // 8192 нейрона в скрытом слое
            nn.AddLayer(new DenseLayer(8192, 16384, Tensor.ApplySwishStatic, Tensor.ApplySwishPrimeStatic)); // 16384 нейрона в скрытом слое
            nn.AddLayer(new DenseLayer(16384, 1, Tensor.ApplyIdentityStatic, Tensor.ApplyIdentityStatic)); // Выходной слой с 1 нейроном

            float learningRate = 0.0001f;
            float decayRate = 0.95f;
            var epochs = 900; // Количество эпох для обучения

            // Подготовка данных для обучения (входные данные и их сумма как выход)
            var trainingData = new List<(Tensor input, Tensor output)>();

            // Генерация 100 примеров сложения (увеличено количество данных для обучения)
            for (int i = 0; i < 1000; i++)
            {
                var input = new Tensor(new int[] { 1, 2 }, new float[] { i + 1, (i + 1) * 2 });
                var output_с = new Tensor(new int[] { 1, 1 }, new float[] { input[0, 0] + input[0, 1] });
                trainingData.Add((input, output_с));
            }

            // Тренировка сети
            for (int i = 0; i < epochs; i++)
            {
                foreach (var data in trainingData)
                {
                    nn.Train(data.input, data.output, learningRate);
                }
                learningRate *= decayRate; // Обновляем learningRate

                if (i % 100 == 0) // Каждые 100 эпох выводим прогресс
                {
                    Console.WriteLine($"Эпоха {i}, выходные данные: " + string.Join("; ", nn.Predict(new Tensor(new int[] { 1, 2 }, new float[] { 5f, 10f })).Data.ToList()));
                }
            }

            // Прогнозируем результат на примере
            var testInput = new Tensor(new int[] { 1, 2 }, new float[] { 5f, 10f });
            var output = nn.Predict(testInput);

            // Вывод данных из тензора как строки
            Console.WriteLine("Прогнозируем результат: " + string.Join("; ", output.Data.ToList()));
        }
    }
}
