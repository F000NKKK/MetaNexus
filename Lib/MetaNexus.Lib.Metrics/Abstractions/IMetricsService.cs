using MetaNexus.Lib.Metrics.Models;

namespace MetaNexus.Lib.Metrics.Abstractions
{
    public interface IMetricsService
    {
        void Submit(RawMetric metric);
    }
}