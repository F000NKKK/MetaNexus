using MetaNexus.Lib.Metrics.Services.Abstractions;
using System.Diagnostics.Metrics;

namespace MetaNexus.Lib.Metrics.Services
{
    internal class MetricsService : IMetricsService
    {
        private readonly Meter _meter;

        public MetricsService(Meter meter)
        {
            _meter = meter;
        }

#pragma warning disable CS8620
        public void SubmitMetric(Models.Metric metric)
        {
            var tagList = metric.Labels?.Select(label => new KeyValuePair<string, object>(label.Key, label.Value)).ToArray() ?? Array.Empty<KeyValuePair<string, object>>();

            switch (metric.Type)
            {
                case Models.MetricType.Counter:
                    var counter = _meter.CreateCounter<double>(metric.Name);
                    counter.Add(metric.Value, tagList);
                    break;

                case Models.MetricType.Gauge:
                    var gauge = _meter.CreateGauge<double>(metric.Name);
                    gauge.Record(metric.Value, tagList);
                    break;

                case Models.MetricType.Histogram:
                    var histogram = _meter.CreateHistogram<double>(metric.Name);
                    histogram.Record(metric.Value, tagList);
                    break;
            }
        }
    }
}