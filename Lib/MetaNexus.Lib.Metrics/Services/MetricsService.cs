using MetaNexus.Lib.Metrics.Abstractions;
using MetaNexus.Lib.Metrics.Models;

namespace MetaNexus.Lib.Metrics.Services
{
    /// <summary>
    /// Сервис для работы с метриками.
    /// </summary>
    internal class MetricsService : IMetricsService
    {
        private readonly IMetricsHost _metricsHost;

        /// <summary>
        /// Инициализирует новый экземпляр сервиса метрик.
        /// </summary>
        /// <param name="metricsHost">Объект, предоставляющий доступ к метрикам.</param>
        public MetricsService(IMetricsHost metricsHost)
        {
            _metricsHost = metricsHost ?? throw new ArgumentNullException(nameof(metricsHost));
        }

        /// <summary>
        /// Приватный метод для обработки метрики.
        /// </summary>
        /// <param name="metricName">Название метрики.</param>
        /// <param name="value">Значение метрики.</param>
        /// <param name="labels">Метки метрики.</param>
        private void ProcessMetricInternal(MetricTypes metricType, string metricName, double value = 0, IEnumerable<KeyValuePair<string, object?>> labels = null)
        {
            var tags = labels.ToArray() ?? Enumerable.Empty<KeyValuePair<string, object?>>().ToArray();

            if (metricType == MetricTypes.Counter)
            {
                _metricsHost.GetCounter(metricName).Add((long)value, tags);
            }
            else if (metricType == MetricTypes.Gauge)
            {
                _metricsHost.GetGauge(metricName).Record(value, tags);
            }
            else if (metricType == MetricTypes.Histogram)
            {
                _metricsHost.GetHistogram(metricName).Record(value, tags);
            }
            else
            {
                throw new InvalidOperationException($"Unsupported metric type for '{metricName}' with type '{metricType.ToString()}'.");
            }
        }



        /// <summary>
        /// Обрабатывает метрику, переданную через объект <see cref="RawMetric"/>.
        /// </summary>
        /// <param name="metric">Объект метрики для обработки.</param>
        public void Submit(RawMetric metric)
        {
            if (metric == null)
                throw new ArgumentNullException(nameof(metric), "Metric cannot be null.");

            var labels = metric.Labels != null
                ? metric.Labels.Select(label => new KeyValuePair<string, object?>(label.Key, label.Value))
                : Enumerable.Empty<KeyValuePair<string, object?>>();

            try
            {
                ProcessMetricInternal(metric.MetricType, metric.Name, metric.Value, labels);
            }
            catch (InvalidOperationException ex)
            {
                throw new InvalidOperationException($"Error processing metric '{metric.Name}'. Ensure the metric type is supported.", ex);
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"An error occurred while processing metric '{metric.Name}'.", ex);
            }
        }
    }
}
