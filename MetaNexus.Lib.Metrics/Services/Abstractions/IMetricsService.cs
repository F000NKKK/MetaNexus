using MetaNexus.Lib.Metrics.Models;

namespace MetaNexus.Lib.Metrics.Services.Abstractions
{
    public interface IMetricsService
    {
        void SubmitMetric(string name, double value, IEnumerable<MetricLabel> labels = null);
    }
}
