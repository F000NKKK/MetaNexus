using MetaNexus.Lib.NeuralNetwork.Math.Optimizers.Abstractions;

namespace MetaNexus.Lib.NeuralNetwork.Math.Optimizers
{
    public class SGD : Optimizer
    {
        public SGD(double learningRate) : base(learningRate) { }

        /// <summary>
        /// Обновление параметров с использованием SGD.
        /// </summary>
        public override void Update(double[] parameters, double[] gradients)
        {
            var updatedParameters = parameters
                .Zip(gradients, (param, grad) => param - LearningRate * grad)
                .ToArray();

            Array.Copy(updatedParameters, parameters, parameters.Length);
        }
    }
}