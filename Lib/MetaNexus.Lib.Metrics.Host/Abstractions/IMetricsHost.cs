using MetaNexus.Lib.Metrics.Models;

namespace MetaNexus.Lib.Metrics.Host.Abstractions
{
    public interface IMetricsHost
    {
        Metric GetMetric(string name, double value, IEnumerable<MetricLabel>? labels = null);
    }
}