using Autofac;
using MetaNexus.App.DemoApp.Abstractions;

namespace MetaNexus.App.DemoApp.Modules.NeuralNetwork
{
    public class IoCModule : Module
    {
        protected override void Load(ContainerBuilder builder)
        {
            // Регистрируем типы
            builder.RegisterType<SimpleNeuralNetwork>().AsSelf().SingleInstance();
            builder.RegisterType<NeuralNetworkTrainingService>().As<INeuralNetworkTrainingService>().SingleInstance();
        }
    }
}
