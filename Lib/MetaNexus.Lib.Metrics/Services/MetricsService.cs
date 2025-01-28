using MetaNexus.Lib.Metrics.Abstractions;
using MetaNexus.Lib.Metrics.Models;
using System.Diagnostics.Metrics;

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
        private void ProcessMetricInternal(string metricName, double value = 0, IEnumerable<KeyValuePair<string, object>> labels = null)
        {
            var metric = _metricsHost.GetMetric<Instrument>(metricName);

            var tags = labels ?? Enumerable.Empty<KeyValuePair<string, object>>();

            switch (metric)
            {
                case Counter<long> counter:
                    counter.Add((long)value, tags.ToArray());
                    break;

                case Gauge<double> gauge:
                    gauge.Record(value, tags.ToArray());
                    break;

                case Histogram<double> histogram:
                    histogram.Record(value, tags.ToArray());
                    break;

                default:
                    throw new InvalidOperationException($"Unsupported metric type for '{metricName}'.");
            }
        }

        /// <summary>
        /// Обрабатывает метрику, переданную через объект <see cref="RawMetric"/>.
        /// </summary>
        /// <param name="metric">Объект метрики для обработки.</param>
        public void ProcessMetric(RawMetric metric)
        {
            if (metric == null)
                throw new ArgumentNullException(nameof(metric), "Metric cannot be null.");

            string metricName = metric.Name;
            double metricValue = metric.Value;

            var labels = metric.Labels != null
                ? metric.Labels.Select(label => new KeyValuePair<string, object>(label.Key, label.Value))
                : Enumerable.Empty<KeyValuePair<string, object>>();

            try
            {
                ProcessMetricInternal(metricName, metricValue, labels);
            }
            catch (InvalidOperationException ex)
            {
                throw new InvalidOperationException($"Error processing metric '{metricName}'. Ensure the metric type is supported.", ex);
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"An error occurred while processing metric '{metricName}'.", ex);
            }
        }
    }
}
