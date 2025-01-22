using Autofac;
using MetaNexus.Lib.Metrics.Configurations;
using MetaNexus.Lib.Metrics.Host;
using MetaNexus.Lib.Metrics.Host.Abstractions;
using MetaNexus.Lib.Metrics.Services;
using MetaNexus.Lib.Metrics.Services.Abstractions;
using System.Diagnostics.Metrics;

namespace MetaNexus.Lib.Metrics
{
    public class IoCModule : Module
    {
        const string meterName = "MetaNexusMetrics";
        const string otlpEndpoint = "http://localhost:4317";
        protected override void Load(ContainerBuilder builder)
        {
            builder.RegisterInstance(MetricsConfiguration.ConfigureMeter(meterName, otlpEndpoint))
                .As<Meter>()
                .SingleInstance();

            builder.RegisterType<MetricsHost>()
                .As<IMetricsHost>()
                .SingleInstance();

            builder.RegisterType<MetricsService>()
                .As<IMetricsService>()
                .SingleInstance();
        }
    }
}
