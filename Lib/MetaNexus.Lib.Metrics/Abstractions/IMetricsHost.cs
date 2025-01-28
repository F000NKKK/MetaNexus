using System.Diagnostics.Metrics;

namespace MetaNexus.Lib.Metrics.Abstractions
{
    internal interface IMetricsHost
    {
        T GetMetric<T>(string metricName) where T : Instrument;
    }
}