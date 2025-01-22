using OpenTelemetry;
using OpenTelemetry.Metrics;
using System.Diagnostics.Metrics;

namespace MetaNexus.Lib.Metrics.Configurations
{
    internal static class MetricsConfiguration
    {
        public static Meter ConfigureMeter(string meterName, string otlpEndpoint)
        {
            var meter = new Meter(meterName);

            Sdk.CreateMeterProviderBuilder()
                .AddMeter(meterName)
                .AddOtlpExporter(options =>
                {
                    options.Endpoint = new Uri(otlpEndpoint);
                })
                .Build();

            return meter;
        }
    }
}
