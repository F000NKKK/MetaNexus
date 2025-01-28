using MetaNexus.App.DemoApp.Abstractions;
using MetaNexus.Lib.Metrics.Abstractions;
using MetaNexus.Lib.Metrics.Consts;
using MetaNexus.Lib.Metrics.Models;
using MetaNexus.Lib.NeuralNetwork.Tensors;
using OpenTelemetry.Metrics;
using System.Diagnostics.Metrics;

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

        public async void TrainAndPredict()
        {
            _metricsService.Submit(new RawMetric(MetricTypes.Gauge, MetricsNames.NEURAL_NETWORK_SERVICE_RUNS_TOTAL, new List<RawMetricLabel>()
            {
                new RawMetricLabel("environment-os-version", $"{Environment.OSVersion}")
            }, ONE_RUN_METRIC_VALUE));

            // Обучение сети
            Console.WriteLine("Начало обучения...");
            _simpleNeuralNetwork.Train();
            Console.WriteLine("Обучение завершено.");

            // Прогнозируем результат на примере
            var testInput = new Tensor(new int[] { 1, 2 }, new float[] { 5f, 10f });
            var output = _simpleNeuralNetwork.Predict(testInput);

            // Вывод данных из тензора как строки
            Console.WriteLine($"Прогнозируем результат для ввода [5, 10]: {string.Join("; ", output.Data.Span.ToArray())}");
        }

        private const int ONE_RUN_METRIC_VALUE = 1;
    }
}
