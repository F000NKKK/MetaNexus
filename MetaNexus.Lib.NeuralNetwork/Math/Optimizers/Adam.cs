using MetaNexus.Lib.NeuralNetwork.Math.Optimizers.Abstractions;

namespace MetaNexus.Lib.NeuralNetwork.Math.Optimizers
{
    public class Adam : Optimizer
    {
        private readonly float _beta1;
        private readonly float _beta2;
        private readonly float _epsilon;
        private float[] _m;
        private float[] _v;
        private int _t;

        public Adam(float learningRate, int parameterSize, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f)
            : base(learningRate)
        {
            _beta1 = beta1;
            _beta2 = beta2;
            _epsilon = epsilon;
            _m = new float[parameterSize];
            _v = new float[parameterSize];
            _t = 0;
        }

        /// <summary>
        /// Обновление параметров с использованием Adam.
        /// </summary>
        public override void Update(float[] parameters, float[] gradients)
        {
            _t++;

            _m = _m
                .Zip(gradients, (m, grad) => _beta1 * m + (1 - _beta1) * grad)
                .ToArray();

            _v = _v
                .Zip(gradients, (v, grad) => _beta2 * v + (1 - _beta2) * grad * grad)
                .ToArray();

            var mCorrected = _m
                .Select(m => m / (1 - MathF.Pow(_beta1, _t)))
                .ToArray();

            var vCorrected = _v
                .Select(v => v / (1 - MathF.Pow(_beta2, _t)))
                .ToArray();

            var updatedParameters = parameters
                .Zip(mCorrected.Zip(vCorrected, (mc, vc) => (mc, vc)),
                    (param, corrected) => param - LearningRate * corrected.mc / (MathF.Sqrt(corrected.vc) + _epsilon))
                .ToArray();

            Array.Copy(updatedParameters, parameters, parameters.Length);
        }
    }
}