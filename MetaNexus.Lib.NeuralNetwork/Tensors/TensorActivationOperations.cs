using MetaNexus.Lib.NeuralNetwork.Tensors.Abstractions;

namespace MetaNexus.Lib.NeuralNetwork.Tensors
{
    public partial struct Tensor : ITensorActivationOperations
    {
        public Tensor ApplyReLU()
        {
            return Apply(x => x > 0 ? x : 0);
        }

        public Tensor ApplyLeakyReLU(float alpha)
        {
            return Apply(x => x > 0 ? x : alpha * x);
        }

        public Tensor ApplySigmoid()
        {
            return Apply(x => 1f / (1f + MathF.Exp(-x)));
        }

        public Tensor ApplyTanh()
        {
            return Apply(x => MathF.Tanh(x));
        }

        public Tensor ApplySoftmax()
        {
            var expValues = _data.Select(x => MathF.Exp(x));
            float sum = expValues.Sum();
            return new Tensor(_shape, expValues.Select(x => x / sum).ToArray());
        }

        public Tensor ApplySwish()
        {
            return Apply(x => x * (1f / (1f + MathF.Exp(-x))));
        }

        public Tensor ApplyELU(float alpha)
        {
            return Apply(x => x >= 0 ? x : alpha * (MathF.Exp(x) - 1f));
        }

        public Tensor ApplySoftplus()
        {
            return Apply(x => MathF.Log(1f + MathF.Exp(x)));
        }

        public Tensor ApplyHardSigmoid()
        {
            return Apply(x => MathF.Min(MathF.Max((x + 1f) / 2f, 0f), 1f));
        }

        public Tensor ApplyGELU()
        {
            return Apply(x => x * (1f + MathF.Tanh(MathF.Sqrt(2f / MathF.PI) * (x + 0.5f))) / 2f);
        }

        public Tensor ApplyHardTanh()
        {
            return Apply(x => MathF.Min(MathF.Max(x, -1f), 1f));
        }

        public Tensor ApplyMish()
        {
            return Apply(x => x * MathF.Tanh(MathF.Log(1f + MathF.Exp(x))));
        }

        public Tensor ApplySigmoidPrime()
        {
            return Apply(x => (1f - x) * x);
        }

        public Tensor ApplyTanhPrime()
        {
            return Apply(x => 1f - x * x);
        }

        public Tensor ApplyIdentity()
        {
            return this;
        }

        public Tensor ApplySoftSign()
        {
            return Apply(x => x / (1f + MathF.Abs(x)));
        }

        public Tensor ApplySwishPrime()
        {
            return Apply(x => 1f / (1f + MathF.Exp(-x)) + x * (MathF.Exp(-x) / (1f + MathF.Exp(-x))));
        }

        public Tensor ApplyGELUPrime()
        {
            return Apply(x => 0.5f * (1f + MathF.Tanh(MathF.Sqrt(2f / MathF.PI) * (x + 0.5f))) + x * 0.5f);
        }
    }
}
