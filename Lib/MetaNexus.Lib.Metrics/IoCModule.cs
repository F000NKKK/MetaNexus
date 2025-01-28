using Autofac;
using MetaNexus.Lib.Metrics.Abstractions;
using MetaNexus.Lib.Metrics.Host.Abstractions;
using MetaNexus.Lib.Metrics.Services;
using MetaNexus.Lib.Metrics.Services.Abstractions;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Hosting;
using OpenTelemetry;
using OpenTelemetry.Exporter;
using OpenTelemetry.Metrics;
using OpenTelemetry.Resources;
using OpenTelemetry.Trace;
using System;
using System.Collections.Generic;

namespace MetaNexus.Lib.Metrics
{
    public class IoCModule : Module
    {
        protected override void Load(ContainerBuilder builder)
        {
            builder.Register(c =>
            {
                var env = c.Resolve<IHostEnvironment>();
                var config = c.Resolve<IConfiguration>();

                // Получаем имя хоста и endpoint для OTLP
                string metricsHostName = env.ApplicationName;
                string otlpEndpoint = config["OpenTelemetry:Endpoint"] ?? "http://localhost:4317";

                // Регистрируем и возвращаем экземпляр MetricsHost
                var metricsHost = new MetricsHost(metricsHostName, typeof(IoCModule).Assembly.GetName().Version?.ToString());
                return metricsHost;
            }).As<IMetricsHost>().SingleInstance();

            builder.RegisterType<MetricsService>()
                   .As<IMetricsService>()
                   .SingleInstance();

            // Создание и регистрация TracerProvider для OpenTelemetry
            builder.Register(c =>
            {
                var env = c.Resolve<IHostEnvironment>();
                var config = c.Resolve<IConfiguration>();
                string metricsHostName = env.ApplicationName;
                string otlpEndpoint = config["OpenTelemetry:Endpoint"] ?? "http://localhost:4317";

                var resourceBuilder = ResourceBuilder.CreateDefault()
                    .AddService(metricsHostName)
                    .AddAttributes(new List<KeyValuePair<string, object>>
                    {
                        new KeyValuePair<string, object>("env", env.EnvironmentName)
                    });

                // Регистрация TracerProvider
                return Sdk.CreateTracerProviderBuilder()
                    .SetResourceBuilder(resourceBuilder)
                    .SetErrorStatusOnException()
                    .AddSource(metricsHostName)
                    .AddAspNetCoreInstrumentation() // Инструментируем ASP.NET Core
                    .AddOtlpExporter(opt =>
                    {
                        opt.Protocol = OtlpExportProtocol.Grpc;
                        opt.Endpoint = new Uri(otlpEndpoint);
                    })
                    .Build();
            }).SingleInstance();

            // Создание и регистрация MeterProvider для OpenTelemetry
            builder.Register(c =>
            {
                var env = c.Resolve<IHostEnvironment>();
                var config = c.Resolve<IConfiguration>();
                string metricsHostName = env.ApplicationName;
                string otlpEndpoint = config["OpenTelemetry:Endpoint"] ?? "http://localhost:4317";

                // Регистрация MeterProvider
                return Sdk.CreateMeterProviderBuilder()
                    .AddMeter(metricsHostName)
                    .AddAspNetCoreInstrumentation() // Инструментируем ASP.NET Core
                    .AddOtlpExporter(opt =>
                    {
                        opt.Protocol = OtlpExportProtocol.Grpc;
                        opt.Endpoint = new Uri(otlpEndpoint);
                    })
                    .Build();
            }).SingleInstance();
        }
    }
}
