using MetaNexus.Lib.NeuralNetwork.Tensors;

namespace SimpleNeuralNetwork
{
    class Program
    {
        static NeuralNetwork nn = new NeuralNetwork();

        static void Main(string[] args)
        {
            // Архитектура сети
            nn.AddLayer(new InputLayer(3, 3, Tensor.ApplyIdentityStatic, Tensor.ApplyIdentityStatic));
            nn.AddLayer(new DenseLayer(3, 3, Tensor.ApplySwishStatic, Tensor.ApplySwishPrimeStatic));
            nn.AddLayer(new DenseLayer(3, 3, Tensor.ApplySigmoidStatic, Tensor.ApplySigmoidPrimeStatic));
            nn.AddLayer(new DenseLayer(3, 1, Tensor.ApplySoftplusStatic, Tensor.ApplySoftplusPrimeStatic));

            float learningRate = 0.001f;
            float decayRate = 0.9f;

            var epochs = 1;

            // Подготовка данных для обучения (входные данные и их зеркальные версии)
            var trainingData = new List<(Tensor input, Tensor output)>
            {
                (new Tensor(new int[] { 1, 3 }, new float[] { 1f, 2f, 3f }), new Tensor(new int[] { 1, 1 }, new float[] { 6f })),
                (new Tensor(new int[] { 1, 3 }, new float[] { 4f, 5f, 6f }), new Tensor(new int[] { 1, 1 }, new float[] { 15f })),
                (new Tensor(new int[] { 1, 3 }, new float[] { 7f, 8f, 9f }), new Tensor(new int[] { 1, 1 }, new float[] { 24f })),
                // добавьте больше пар данных
            };

            // Тренировка сети с многопоточностью
            for (int i = 0; i < epochs; i++)
            {
                foreach (var data in trainingData)
                {
                    nn.Train(data.input, data.output, learningRate);
                }
                learningRate *= decayRate; // Обновляем learningRate
                Console.WriteLine("Выходные данные: " + string.Join("; ", nn.Predict(new Tensor(new int[] { 1, 3 }, new float[] { 3f, 2f, 1f })).Data.ToList()));
            }

            // Прогнозируем результат
            var output = nn.Predict(new Tensor(new int[] { 1, 3 }, new float[] { 3f, 2f, 1f }));
            // Вывод данных из тензора как строки
            Console.WriteLine("Выходные данные: " + string.Join("; ", output.Data.ToList()));
        }
    }
}
