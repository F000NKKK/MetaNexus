﻿using Autofac;
using MetaNexus.App.DemoApp.Abstractions;

namespace MetaNexus.App.DemoApp
{
    class Program
    {
        static void Main(string[] args)
        {
            var builder = new ContainerBuilder();

            builder.RegisterModule<MetaNexus.App.DemoApp.Modules.NeuralNetwork.IoCModule>();
            builder.RegisterModule<MetaNexus.Lib.Metrics.IoCModule>();

            var container = builder.Build();

            var trainingService = container.Resolve<INeuralNetworkTrainingService>();

            trainingService.TrainAndPredict();

            Console.ReadLine();
        }
    }
}
