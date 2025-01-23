using Autofac;
using MetaNexus.Lib.Metrics.Configurations;
using MetaNexus.Lib.Metrics.Host;
using MetaNexus.Lib.Metrics.Host.Abstractions;
using MetaNexus.Lib.Metrics.Services;
using MetaNexus.Lib.Metrics.Services.Abstractions;
using Microsoft.Extensions.Logging;
using OpenTelemetry.Metrics;
using System.Diagnostics.Metrics;

namespace MetaNexus.Lib.Metrics
{
    public class IoCModule : Module
    {
        const string meterName = "MetaNexusMetrics";
        const string otlpEndpointHttp = "http://localhost:4318";

        protected override void Load(ContainerBuilder builder)
        {
            builder.RegisterInstance(MetricsConfiguration.ConfigureMeterProvider(meterName, otlpEndpointHttp))
                .As<MeterProvider>()
                .SingleInstance();

            builder.Register(c =>
            {
                return new Meter(meterName);
            })
            .As<Meter>()
            .SingleInstance();

            builder.RegisterType<MetricsHost>()
                .As<IMetricsHost>()
                .SingleInstance();

            builder.RegisterType<MetricsService>()
                .As<IMetricsService>()
                .SingleInstance();

            builder.Register(c =>
            {
                var loggerFactory = LoggerFactory.Create(builder => builder.AddConsole());
                return loggerFactory.CreateLogger<MetricsService>();
            }).As<ILogger<MetricsService>>()
              .SingleInstance();
        }
    }
}
