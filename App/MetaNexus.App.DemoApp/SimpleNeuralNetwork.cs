using MetaNexus.Lib.Metrics.Abstractions;
using MetaNexus.Lib.NeuralNetwork.Tensors;

namespace MetaNexus.App.DemoApp
{
    public class SimpleNeuralNetwork
    {
        private NeuralNetwork nn;

        public SimpleNeuralNetwork(IMetricsService metricsService)
        {
            nn = new NeuralNetwork(metricsService);

            // Архитектура сети с уменьшенными слоями, чтобы избежать переполнения
            nn.AddLayer(new InputLayer(2, 2, Tensor.ApplyIdentityStatic, Tensor.ApplyIdentityStatic)); // 2 нейрона во входном слое
            nn.AddLayer(new DenseLayer(2, 512, Tensor.ApplyIdentityStatic, Tensor.ApplyIdentityStatic)); // 512 нейронов во входном слое
            nn.AddLayer(new DenseLayer(512, 1024, Tensor.ApplySwishStatic, Tensor.ApplySwishPrimeStatic)); // 1024 нейрона в скрытом слое
            nn.AddLayer(new DenseLayer(1024, 1, Tensor.ApplyIdentityStatic, Tensor.ApplyIdentityStatic)); // Выходной слой с 1 нейроном
        }

        public void Train(int epochs = 900, float learningRate = 0.1f, float decayRate = 0.95f)
        {
            var trainingData = GenerateTrainingData();

            for (int i = 0; i < epochs; i++)
            {
                foreach (var data in trainingData)
                {
                    nn.Train(data.input, data.output, learningRate);
                }

                learningRate *= decayRate; // Обновляем learningRate

                if (i % 100 == 0) // Каждые 100 эпох выводим прогресс
                {
                    Console.WriteLine($"Эпоха {i}, выходные данные: " + string.Join("; ", nn.Predict(new Tensor(new int[] { 1, 2 }, new float[] { 5f, 10f })).Data.Span.ToArray()));
                }
            }
        }

        public Tensor Predict(Tensor input)
        {
            return nn.Predict(input);
        }

        private List<(Tensor input, Tensor output)> GenerateTrainingData()
        {
            var trainingData = new List<(Tensor input, Tensor output)>();

            // Генерация 1000 примеров сложения
            for (int i = 0; i < 1000; i++)
            {
                var input = new Tensor(new int[] { 1, 2 }, new float[] { i + 1, (i + 1) * 2 });
                var output_с = new Tensor(new int[] { 1, 1 }, new float[] { input[0, 0] + input[0, 1] });
                trainingData.Add((input, output_с));
            }

            return trainingData;
        }
    }
}
