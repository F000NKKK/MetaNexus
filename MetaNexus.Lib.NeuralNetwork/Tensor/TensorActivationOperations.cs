using MetaNexus.Lib.NeuralNetwork.Tensor.Abstractions;

namespace MetaNexus.Lib.NeuralNetwork.Tensor
{
    public partial struct Tensor : ITensorActivationOperations
    {
        // Применение функции ReLU
        public Tensor ApplyReLU()
        {
            return Apply(x => x > 0 ? x : 0);
        }

        // Применение Leaky ReLU
        public Tensor ApplyLeakyReLU(float alpha)
        {
            return Apply(x => x > 0 ? x : alpha * x);
        }

        // Применение Sigmoid
        public Tensor ApplySigmoid()
        {
            return Apply(x => 1f / (1f + MathF.Exp(-x)));
        }

        // Применение Tanh
        public Tensor ApplyTanh()
        {
            return Apply(x => MathF.Tanh(x));
        }

        // Применение Softmax
        public Tensor ApplySoftmax()
        {
            var expValues = _data.Select(x => MathF.Exp(x));
            float sum = expValues.Sum();
            return new Tensor(_shape, expValues.Select(x => x / sum).ToArray());
        }

        // Применение Swish
        public Tensor ApplySwish()
        {
            return Apply(x => x * (1f / (1f + MathF.Exp(-x))));
        }

        // Применение ELU
        public Tensor ApplyELU(float alpha)
        {
            return Apply(x => x >= 0 ? x : alpha * (MathF.Exp(x) - 1f));
        }

        // Применение Softplus
        public Tensor ApplySoftplus()
        {
            return Apply(x => MathF.Log(1f + MathF.Exp(x)));
        }

        // Применение HardSigmoid
        public Tensor ApplyHardSigmoid()
        {
            return Apply(x => MathF.Min(MathF.Max((x + 1f) / 2f, 0f), 1f));
        }

        // Применение GELU
        public Tensor ApplyGELU()
        {
            return Apply(x => x * (1f + MathF.Tanh(MathF.Sqrt(2f / MathF.PI) * (x + 0.5f))) / 2f);
        }

        // Применение HardTanh
        public Tensor ApplyHardTanh()
        {
            return Apply(x => MathF.Min(MathF.Max(x, -1f), 1f));
        }

        // Применение Mish
        public Tensor ApplyMish()
        {
            return Apply(x => x * MathF.Tanh(MathF.Log(1f + MathF.Exp(x))));
        }

        // Применение Sigmoid Prime
        public Tensor ApplySigmoidPrime()
        {
            return Apply(x => (1f - x) * x);
        }

        // Применение Tanh Prime
        public Tensor ApplyTanhPrime()
        {
            return Apply(x => 1f - x * x);
        }

        // Применение Identity
        public Tensor ApplyIdentity()
        {
            return this;
        }

        // Применение SoftSign
        public Tensor ApplySoftSign()
        {
            return Apply(x => x / (1f + MathF.Abs(x)));
        }

        // Применение Swish Prime
        public Tensor ApplySwishPrime()
        {
            return Apply(x => 1f / (1f + MathF.Exp(-x)) + x * (MathF.Exp(-x) / (1f + MathF.Exp(-x))));
        }

        // Применение GELU Prime
        public Tensor ApplyGELUPrime()
        {
            return Apply(x => 0.5f * (1f + MathF.Tanh(MathF.Sqrt(2f / MathF.PI) * (x + 0.5f))) + x * 0.5f);
        }
    }
}
