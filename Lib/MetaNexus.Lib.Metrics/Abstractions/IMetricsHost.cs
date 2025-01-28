using System.Diagnostics.Metrics;

namespace MetaNexus.Lib.Metrics.Abstractions
{
    internal interface IMetricsHost
    {
        Counter<double> GetCounter(string metricName);
        Gauge<double> GetGauge(string metricName);
        Histogram<double> GetHistogram(string metricName);
    }
}