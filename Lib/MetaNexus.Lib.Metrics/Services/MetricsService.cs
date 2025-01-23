using MetaNexus.Lib.Metrics.Host.Abstractions;
using MetaNexus.Lib.Metrics.Models;
using MetaNexus.Lib.Metrics.Services.Abstractions;
using Microsoft.Extensions.Logging;
using OpenTelemetry.Metrics;
using System.Diagnostics.Metrics;

namespace MetaNexus.Lib.Metrics.Services
{
    internal class MetricsService : IMetricsService
    {
        private readonly Meter _meter;
        private readonly IMetricsHost _metricsHost;
        private readonly ILogger<MetricsService> _logger;
        private readonly Dictionary<string, object> _metricCache;
        private readonly MeterProvider _meterProvider;

        public MetricsService(Meter meter, IMetricsHost metricsHost, ILogger<MetricsService> logger, MeterProvider meterProvider)
        {
            _meter = meter;
            _metricsHost = metricsHost;
            _logger = logger;
            _metricCache = new Dictionary<string, object>();
            _meterProvider = meterProvider;
        }

        public void SubmitMetric(string name, double value, IEnumerable<MetricLabel>? labels = null)
        {
            try
            {
                var metric = _metricsHost.GetMetric(name, value, labels);

                var tagList = metric.Labels?.Select(label => new KeyValuePair<string, object>(label.Key, label.Value)).ToArray() ?? Array.Empty<KeyValuePair<string, object>>();

                _logger.LogInformation($"Submitting metric: {metric.Name}, Value: {metric.Value}, Labels: {string.Join(", ", tagList)}");

                if (!_metricCache.TryGetValue(metric.Name, out var registeredMetric))
                {
                    registeredMetric = metric.Type switch
                    {
                        Models.MetricType.Counter => _meter.CreateCounter<double>(metric.Name),
                        Models.MetricType.Gauge => _meter.CreateGauge<double>(metric.Name),
                        Models.MetricType.Histogram => _meter.CreateHistogram<double>(metric.Name),
                        _ => throw new InvalidOperationException($"Unsupported metric type: {metric.Type}")
                    };

                    _metricCache[metric.Name] = registeredMetric;
                }

#pragma warning disable CS8620
                switch (metric.Type)
                {
                    case Models.MetricType.Counter:
                        ((Counter<double>)registeredMetric).Add(metric.Value, tagList);
                        break;

                    case Models.MetricType.Gauge:
                        ((Gauge<double>)registeredMetric).Record(metric.Value, tagList);
                        break;

                    case Models.MetricType.Histogram:
                        ((Histogram<double>)registeredMetric).Record(metric.Value, tagList);
                        break;

                    default:
                        throw new InvalidOperationException($"Unsupported metric type: {metric.Type}");
                }
#pragma warning restore CS8620
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error submitting metric");
            }
        }
    }
}
