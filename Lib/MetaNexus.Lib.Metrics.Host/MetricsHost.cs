using MetaNexus.Lib.Metrics.Host.Abstractions;
using MetaNexus.Lib.Metrics.Models;

namespace MetaNexus.Lib.Metrics.Host
{
    public class MetricsHost : IMetricsHost
    {
        /// <summary>
        /// Возвращает объект метрики, готовый к использованию
        /// </summary>
        /// <param name="name">Имя метрики</param>
        /// <param name="value">Значение метрики</param>
        /// <param name="labels">Список меток (тегов)</param>
        /// <returns>Объект Metric</returns>
        public Metric GetMetric(string name, double value, IEnumerable<MetricLabel>? labels = null)
        {
            return name switch
            {
                MetricNames.REQUEST_COUNT => CreateMetric(name, MetricType.Counter, value, labels),
                MetricNames.ACTIVE_USERS => CreateMetric(name, MetricType.Gauge, value, labels),
                MetricNames.REQUEST_LATENCY => CreateMetric(name, MetricType.Histogram, value, labels),
                MetricNames.NEURAL_NETWORK_SERVICE_RUNS_TOTAL => CreateMetric(name, MetricType.Counter, value, labels),
                MetricNames.PREDICTS_INPUT_OUTPUT_DATA => CreateMetric(name, MetricType.Counter, value, labels),
                _ => throw new KeyNotFoundException($"Metric with name '{name}' is nor defined in {nameof(MetricNames)}")
            };
        }

        /// <summary>
        /// Создает объект метрики
        /// </summary>
        /// <param name="name">Имя метрики</param>
        /// <param name="type">Тип метрики</param>
        /// <param name="value">Значение метрики</param>
        /// <param name="labels">Список меток (тегов)</param>
        /// <returns>Объект Metric</returns>
        private static Metric CreateMetric(string name, MetricType type, double value, IEnumerable<MetricLabel>? labels = null)
        {
            return new Metric(name, type, labels, value);
        }
    }
}
