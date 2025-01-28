using MetaNexus.Lib.Metrics.Abstractions;
using MetaNexus.Lib.Metrics.Consts;
using OpenTelemetry.Metrics;
using System.Diagnostics.Metrics;

namespace MetaNexus.Lib.Metrics.Services
{
    internal class MetricsHost : IMetricsHost
    {
        private readonly Meter _meter;
        private readonly Dictionary<string, Counter<double>> _counterMetrics;
        private readonly Dictionary<string, Gauge<double>> _gaugeMetrics;
        private readonly Dictionary<string, Histogram<double>> _histogramMetrics;

        public MetricsHost(string metricsHostName, string? version)
        {
            _meter = new Meter(metricsHostName, version: version);
            _counterMetrics = new Dictionary<string, Counter<double>>();
            _gaugeMetrics = new Dictionary<string, Gauge<double>>();
            _histogramMetrics = new Dictionary<string, Histogram<double>>();

            RegisterMetrics();
        }

        private void RegisterMetrics()
        {
            RegisterCounterMetric(MetricsNames.CPU_USAGE, _meter.CreateCounter<double>(MetricsNames.CPU_USAGE));
            RegisterCounterMetric(MetricsNames.REQUEST_COUNT, _meter.CreateCounter<double>(MetricsNames.REQUEST_COUNT));
            RegisterCounterMetric(MetricsNames.ERROR_COUNT, _meter.CreateCounter<double>(MetricsNames.ERROR_COUNT));

            RegisterHistogramMetric(MetricsNames.REQUEST_DURATION, _meter.CreateHistogram<double>(MetricsNames.REQUEST_DURATION));

            RegisterGaugeMetric(MetricsNames.MEMORY_USAGE, _meter.CreateGauge<double>(MetricsNames.MEMORY_USAGE));
            RegisterGaugeMetric(MetricsNames.NEURAL_NETWORK_SERVICE_RUNS_TOTAL, _meter.CreateGauge<double>(MetricsNames.NEURAL_NETWORK_SERVICE_RUNS_TOTAL));
            RegisterGaugeMetric(MetricsNames.PREDICTS_INPUT_OUTPUT_DATA, _meter.CreateGauge<double>(MetricsNames.PREDICTS_INPUT_OUTPUT_DATA));
        }

        private void RegisterCounterMetric(string metricName, Counter<double> metric)
        {
            if (!_counterMetrics.ContainsKey(metricName))
            {
                _counterMetrics[metricName] = metric;
            }
        }

        private void RegisterGaugeMetric(string metricName, Gauge<double> metric)
        {
            if (!_counterMetrics.ContainsKey(metricName))
            {
                _gaugeMetrics[metricName] = metric;
            }
        }

        private void RegisterHistogramMetric(string metricName, Histogram<double> metric)
        {
            if (!_counterMetrics.ContainsKey(metricName))
            {
                _histogramMetrics[metricName] = metric;
            }
        }

        public Counter<double> GetCounter(string metricName)
        {
            if (_counterMetrics.TryGetValue(metricName, out var metric))
            {
                return metric;
            }
            else
            {
                throw new KeyNotFoundException($"Metric with name '{metricName}' not found.");
            }
        }

        public Gauge<double> GetGauge(string metricName)
        {
            if (_gaugeMetrics.TryGetValue(metricName, out var metric))
            {
                return metric;
            }
            else
            {
                throw new KeyNotFoundException($"Metric with name '{metricName}' not found.");
            }
        }

        public Histogram<double> GetHistogram(string metricName)
        {
            if (_histogramMetrics.TryGetValue(metricName, out var metric))
            {
                return metric;
            }
            else
            {
                throw new KeyNotFoundException($"Metric with name '{metricName}' not found.");
            }
        }
    }
}
