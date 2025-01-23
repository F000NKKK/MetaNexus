using MetaNexus.App.DemoApp.Abstractions;
using MetaNexus.Lib.Metrics.Host;
using MetaNexus.Lib.Metrics.Models;
using MetaNexus.Lib.Metrics.Services.Abstractions;
using MetaNexus.Lib.NeuralNetwork.Tensors;

namespace MetaNexus.App.DemoApp
{
    public class NeuralNetworkTrainingService : INeuralNetworkTrainingService
    {
        private readonly SimpleNeuralNetwork _simpleNeuralNetwork;
        private readonly IMetricsService _metricsService;

        public NeuralNetworkTrainingService(SimpleNeuralNetwork simpleNeuralNetwork, IMetricsService metricsService)
        {
            _simpleNeuralNetwork = simpleNeuralNetwork;
            _metricsService = metricsService;
        }

        public void TrainAndPredict()
        {
            _metricsService.SubmitMetric(MetricNames.NEURAL_NETWORK_SERVICE_RUNS_TOTAL, 1, new List<MetricLabel>() { new MetricLabel("environment-os-version", $"{Environment.OSVersion}" ) } );
            // Обучение сети
            Console.WriteLine("Начало обучения...");
            //_simpleNeuralNetwork.Train();
            Console.WriteLine("Обучение завершено.");

            // Прогнозируем результат на примере
            //var testInput = new Tensor(new int[] { 1, 2 }, new float[] { 5f, 10f });
            //var output = _simpleNeuralNetwork.Predict(testInput);

            // Вывод данных из тензора как строки
            //Console.WriteLine($"Прогнозируем результат для ввода [5, 10]: {string.Join("; ", output.Data.Span.ToArray())}");
        }
    }
}
