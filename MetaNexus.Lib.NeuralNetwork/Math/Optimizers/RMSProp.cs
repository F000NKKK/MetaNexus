using MetaNexus.Lib.NeuralNetwork.Math.Optimizers.Abstractions;

namespace MetaNexus.Lib.NeuralNetwork.Math.Optimizers
{
    public class RMSProp : Optimizer
    {
        private readonly float _decayRate;
        private readonly float _epsilon;
        private float[] _cache;

        public RMSProp(float learningRate, int parameterSize, float decayRate = 0.9f, float epsilon = 1e-8f)
            : base(learningRate)
        {
            _decayRate = decayRate;
            _epsilon = epsilon;
            _cache = new float[parameterSize];
        }

        /// <summary>
        /// Обновление параметров с использованием RMSProp.
        /// </summary>
        public override void Update(float[] parameters, float[] gradients)
        {
            _cache = _cache
                .Zip(gradients, (cache, grad) => _decayRate * cache + (1 - _decayRate) * grad * grad)
                .ToArray();

            var updatedParameters = parameters
                .Zip(_cache, (param, cache) => param - LearningRate * gradients[Array.IndexOf(parameters, param)] / (float.Sqrt(cache) + _epsilon))
                .ToArray();

            Array.Copy(updatedParameters, parameters, parameters.Length);
        }
    }
}