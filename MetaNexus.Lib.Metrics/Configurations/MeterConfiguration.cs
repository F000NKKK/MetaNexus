using OpenTelemetry;
using OpenTelemetry.Metrics;
using OpenTelemetry.Exporter;

namespace MetaNexus.Lib.Metrics.Configurations
{
    internal static class MetricsConfiguration
    {
        public static MeterProvider ConfigureMeterProvider(string meterName, string otlpEndpointHttp)
        {
            var meterProvider = Sdk.CreateMeterProviderBuilder()
                .AddMeter(meterName)
                .AddOtlpExporter(options =>
                {
                    options.Endpoint = new Uri(otlpEndpointHttp);
                    options.Protocol = OtlpExportProtocol.HttpProtobuf;
                })
                .Build();

            return meterProvider;
        }
    }
}