using MetaNexus.Lib.Metrics.Abstractions;
using OpenTelemetry.Metrics;
using System.Diagnostics.Metrics;

namespace MetaNexus.Lib.Metrics.Services
{
    internal class MetricsHost : IMetricsHost
    {
        private readonly Meter _meter;
        private readonly Dictionary<string, Instrument> _metrics;

        // Конструктор
        public MetricsHost(string metricsHostName, string? version)
        {
            _meter = new Meter(metricsHostName);  // Здесь создаем Meter напрямую, без использования GetMeter
            _metrics = new Dictionary<string, Instrument>();

            // Инициализируем метрики
            RegisterMetrics();
        }

        // Метод регистрации метрик
        private void RegisterMetrics()
        {
            RegisterMetric(MetricsNames.CPU_USAGE, _meter.CreateCounter<long>(MetricsNames.CPU_USAGE));
            RegisterMetric(MetricsNames.MEMORY_USAGE, _meter.CreateGauge<double>(MetricsNames.MEMORY_USAGE));
            RegisterMetric(MetricsNames.REQUEST_COUNT, _meter.CreateCounter<long>(MetricsNames.REQUEST_COUNT));
            RegisterMetric(MetricsNames.ERROR_COUNT, _meter.CreateCounter<long>(MetricsNames.ERROR_COUNT));
            RegisterMetric(MetricsNames.REQUEST_DURATION, _meter.CreateHistogram<double>(MetricsNames.REQUEST_DURATION));
        }

        // Метод для безопасной регистрации метрики
        private void RegisterMetric(string metricName, Instrument metric)
        {
            if (!_metrics.ContainsKey(metricName))
            {
                _metrics[metricName] = metric;
            }
        }

        // Метод получения метрики по имени через индексатор
        public T GetMetric<T>(string metricName) where T : Instrument
        {
            if (_metrics.TryGetValue(metricName, out var metric))
            {
                // Проверяем, что метрика соответствует ожидаемому типу
                if (metric is T typedMetric)
                {
                    return typedMetric;
                }
                else
                {
                    throw new InvalidCastException($"Metric '{metricName}' is not of type '{typeof(T)}'.");
                }
            }
            else
            {
                throw new KeyNotFoundException($"Metric with name '{metricName}' not found.");
            }
        }
    }
}
