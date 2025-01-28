using MetaNexus.Lib.Metrics.Models;
using OpenTelemetry.Metrics;

namespace MetaNexus.Lib.Metrics.Abstractions
{
    public interface IMetricsService
    {
        void Submit(RawMetric metric);
    }
}