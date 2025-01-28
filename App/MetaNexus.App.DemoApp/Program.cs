using Autofac;
using MetaNexus.App.DemoApp.Abstractions;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Hosting;
using System;
using System.IO;

namespace MetaNexus.App.DemoApp
{
    class Program
    {
        static void Main(string[] args)
        {
            // Создание контейнера Autofac
            var builder = new ContainerBuilder();

            // Создаем и настраиваем IConfiguration
            var configuration = new ConfigurationBuilder()
                .SetBasePath(Directory.GetCurrentDirectory()) // Указываем базовый путь для конфигурации
                .AddJsonFile("appsettings.json", optional: true, reloadOnChange: true) // Добавляем файл конфигурации
                .Build();

            // Регистрируем IConfiguration в контейнере
            builder.RegisterInstance(configuration).As<IConfiguration>().SingleInstance();

            // Регистрируем модули или сервисы в контейнере Autofac
            builder.RegisterModule<MetaNexus.Lib.Metrics.IoCModule>();
            builder.RegisterModule<MetaNexus.App.DemoApp.Modules.NeuralNetwork.IoCModule>();

            // Регистрируем сервисы для интерфейсов
            builder.RegisterType<NeuralNetworkTrainingService>().As<INeuralNetworkTrainingService>().SingleInstance();

            // Строим контейнер
            var container = builder.Build();

            // Разрешаем зависимости через Autofac
            var trainingService = container.Resolve<INeuralNetworkTrainingService>();

            // Запускаем тренировку и прогноз
            trainingService.TrainAndPredict();

            Console.ReadLine();
        }
    }
}
