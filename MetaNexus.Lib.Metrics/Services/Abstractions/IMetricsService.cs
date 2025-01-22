using MetaNexus.Lib.Metrics.Models;

namespace MetaNexus.Lib.Metrics.Services.Abstractions
{
    public interface IMetricsService
    {
        void SubmitMetric(Metric metricEvent);
    }
}
