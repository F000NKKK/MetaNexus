using MetaNexus.Lib.NeuralNetwork.Math.Optimizers.Abstractions;

namespace MetaNexus.Lib.NeuralNetwork.Math.Optimizers
{
    public class RMSProp : Optimizer
    {
        private readonly double _decayRate;
        private readonly double _epsilon;
        private double[] _cache;

        public RMSProp(double learningRate, int parameterSize, double decayRate = 0.9, double epsilon = 1e-8)
            : base(learningRate)
        {
            _decayRate = decayRate;
            _epsilon = epsilon;
            _cache = new double[parameterSize];
        }

        /// <summary>
        /// Обновление параметров с использованием RMSProp.
        /// </summary>
        public override void Update(double[] parameters, double[] gradients)
        {
            _cache = _cache
                .Zip(gradients, (cache, grad) => _decayRate * cache + (1 - _decayRate) * grad * grad)
                .ToArray();

            var updatedParameters = parameters
                .Zip(_cache, (param, cache) => param - LearningRate * gradients[Array.IndexOf(parameters, param)] / (double.Sqrt(cache) + _epsilon))
                .ToArray();

            Array.Copy(updatedParameters, parameters, parameters.Length);
        }
    }
}