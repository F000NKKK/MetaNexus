using OpenTelemetry;
using OpenTelemetry.Metrics;
using OpenTelemetry.Exporter;

namespace MetaNexus.Lib.Metrics.Configurations
{
    internal static class MetricsConfiguration
    {
        public static MeterProvider ConfigureMeterProvider(string meterName, string otlpEndpoint, OtlpExportProtocol otlpExportProtocol)
        {
            var metricReader = new PeriodicExportingMetricReader(new OtlpMetricExporter(new OtlpExporterOptions
            {
                Endpoint = new Uri(otlpEndpoint),
                Protocol = otlpExportProtocol,
                TimeoutMilliseconds = 1000 // Установим таймаут
            }), 2000); // Отправка каждые 2 секунды

            // Создание MeterProvider
            var meterProvider = Sdk.CreateMeterProviderBuilder()
                .AddMeter(meterName)
                .AddReader(metricReader) // Добавляем новый MetricReader
                .Build();

            return meterProvider;
        }
    }
}