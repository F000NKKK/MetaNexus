using MetaNexus.Lib.NeuralNetwork.Math.Optimizers.Abstractions;

namespace MetaNexus.Lib.NeuralNetwork.Math.Optimizers
{
    public class Adam : Optimizer
    {
        private readonly double _beta1;
        private readonly double _beta2;
        private readonly double _epsilon;
        private double[] _m;
        private double[] _v;
        private int _t;

        public Adam(double learningRate, int parameterSize, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8)
            : base(learningRate)
        {
            _beta1 = beta1;
            _beta2 = beta2;
            _epsilon = epsilon;
            _m = new double[parameterSize];
            _v = new double[parameterSize];
            _t = 0;
        }

        /// <summary>
        /// Обновление параметров с использованием Adam.
        /// </summary>
        public override void Update(double[] parameters, double[] gradients)
        {
            _t++;

            _m = _m
                .Zip(gradients, (m, grad) => _beta1 * m + (1 - _beta1) * grad)
                .ToArray();

            _v = _v
                .Zip(gradients, (v, grad) => _beta2 * v + (1 - _beta2) * grad * grad)
                .ToArray();

            var mCorrected = _m
                .Select(m => m / (1 - double.Pow(_beta1, _t)))
                .ToArray();

            var vCorrected = _v
                .Select(v => v / (1 - double.Pow(_beta2, _t)))
                .ToArray();

            var updatedParameters = parameters
                .Zip(mCorrected.Zip(vCorrected, (mc, vc) => (mc, vc)),
                    (param, corrected) => param - LearningRate * corrected.mc / (double.Sqrt(corrected.vc) + _epsilon))
                .ToArray();

            Array.Copy(updatedParameters, parameters, parameters.Length);
        }
    }
}