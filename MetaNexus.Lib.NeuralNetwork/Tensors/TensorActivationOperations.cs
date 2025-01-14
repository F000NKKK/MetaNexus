using MetaNexus.Lib.NeuralNetwork.Tensors.Abstractions;

namespace MetaNexus.Lib.NeuralNetwork.Tensors
{
    public partial struct Tensor : ITensorActivationOperations, ITensorActivationOperationsPrime
    {
        // Нестатические методы для активаций
        Tensor ITensorActivationOperations.ApplyReLU()
        {
            return Apply(x => x > 0 ? x : 0);
        }

        Tensor ITensorActivationOperations.ApplyLeakyReLU(float alpha)
        {
            return Apply(x => x > 0 ? x : alpha * x);
        }

        Tensor ITensorActivationOperations.ApplySigmoid()
        {
            return Apply(x => 1f / (1f + MathF.Exp(-x)));
        }

        Tensor ITensorActivationOperations.ApplyTanh()
        {
            return Apply(x => MathF.Tanh(x));
        }

        Tensor ITensorActivationOperations.ApplySoftmax()
        {
            var expValues = _data.Select(x => MathF.Exp(x));
            float sum = expValues.Sum();
            return new Tensor(_shape, expValues.Select(x => x / sum).ToArray());
        }

        Tensor ITensorActivationOperations.ApplySwish()
        {
            return Apply(x => x * (1f / (1f + MathF.Exp(-x))));
        }

        Tensor ITensorActivationOperations.ApplyELU(float alpha)
        {
            return Apply(x => x >= 0 ? x : alpha * (MathF.Exp(x) - 1f));
        }

        Tensor ITensorActivationOperations.ApplySoftplus()
        {
            return Apply(x => MathF.Log(1f + MathF.Exp(x)));
        }

        Tensor ITensorActivationOperations.ApplyHardSigmoid()
        {
            return Apply(x => MathF.Min(MathF.Max((x + 1f) / 2f, 0f), 1f));
        }

        Tensor ITensorActivationOperations.ApplyGELU()
        {
            return Apply(x => x * (1f + MathF.Tanh(MathF.Sqrt(2f / MathF.PI) * (x + 0.5f))) / 2f);
        }

        Tensor ITensorActivationOperations.ApplyHardTanh()
        {
            return Apply(x => MathF.Min(MathF.Max(x, -1f), 1f));
        }

        Tensor ITensorActivationOperations.ApplyMish()
        {
            return Apply(x => x * MathF.Tanh(MathF.Log(1f + MathF.Exp(x))));
        }

        // Реализация производных функций активации
        Tensor ITensorActivationOperationsPrime.ApplySigmoidPrime()
        {
            return Apply(x =>
            {
                float sigmoid = 1f / (1f + MathF.Exp(-x));
                return sigmoid * (1f - sigmoid);
            });
        }

        Tensor ITensorActivationOperationsPrime.ApplyTanhPrime()
        {
            return Apply(x =>
            {
                float tanh = MathF.Tanh(x);
                return 1f - tanh * tanh;
            });
        }

        Tensor ITensorActivationOperationsPrime.ApplyReLUPrime()
        {
            return Apply(x => x > 0 ? 1f : 0f);
        }

        Tensor ITensorActivationOperationsPrime.ApplyLeakyReLUPrime(float alpha)
        {
            return Apply(x => x > 0 ? 1f : alpha);
        }

        Tensor ITensorActivationOperationsPrime.ApplySoftplusPrime()
        {
            return Apply(x => 1f / (1f + MathF.Exp(-x)));
        }

        Tensor ITensorActivationOperationsPrime.ApplySwishPrime()
        {
            return Apply(x =>
            {
                float sigmoid = 1f / (1f + MathF.Exp(-x));
                return sigmoid + x * sigmoid * (1f - sigmoid);
            });
        }

        Tensor ITensorActivationOperationsPrime.ApplyGELUPrime()
        {
            return Apply(x =>
            {
                float c = 0.0356774f * x * x * x + 0.797885f * x;
                return 0.5f * (1f + MathF.Tanh(c)) + 0.0535161f * x * (1f - MathF.Pow(MathF.Tanh(c), 2));
            });
        }

        Tensor ITensorActivationOperationsPrime.ApplyHardSigmoidPrime()
        {
            return Apply(x => x > -2.5f && x < 2.5f ? 0.2f : 0f);
        }

        Tensor ITensorActivationOperationsPrime.ApplyHardTanhPrime()
        {
            return Apply(x => x > -1f && x < 1f ? 1f : 0f);
        }

        Tensor ITensorActivationOperationsPrime.ApplyMishPrime()
        {
            return Apply(x =>
            {
                float sp = 1f / (1f + MathF.Exp(-x));
                float omega = 4f * (x + 1f) + 4f * MathF.Exp(2f * x) + MathF.Exp(3f * x) + MathF.Exp(x) * (4f * x + 6f);
                float delta = 2f * MathF.Exp(x) + MathF.Exp(2f * x) + 2f;
                return sp + sp * (1f - sp) * omega / (delta * delta);
            });
        }

        Tensor ITensorActivationOperations.ApplyIdentity()
        {
            return this;
        }

        Tensor ITensorActivationOperations.ApplySoftSign()
        {
            return Apply(x => x / (1f + MathF.Abs(x)));
        }

        // Статические версии функций активации
        public static Tensor ApplySoftSignStatic(Tensor tensor)
        {
            return ((ITensorActivationOperations)tensor).ApplySoftSign();
        }
        public static Tensor ApplyIdentityStatic(Tensor tensor)
        {
            return ((ITensorActivationOperations)tensor).ApplyIdentity();
        }

        public static Tensor ApplySigmoidStatic(Tensor tensor)
        {
            return ((ITensorActivationOperations)tensor).ApplySigmoid();
        }

        public static Tensor ApplyTanhStatic(Tensor tensor)
        {
            return ((ITensorActivationOperations)tensor).ApplyTanh();
        }

        public static Tensor ApplyReLUStatic(Tensor tensor)
        {
            return ((ITensorActivationOperations)tensor).ApplyReLU();
        }

        public static Tensor ApplyLeakyReLUStatic(Tensor tensor, float alpha)
        {
            return ((ITensorActivationOperations)tensor).ApplyLeakyReLU(alpha);
        }

        public static Tensor ApplySoftplusStatic(Tensor tensor)
        {
            return ((ITensorActivationOperations)tensor).ApplySoftplus();
        }

        public static Tensor ApplySwishStatic(Tensor tensor)
        {
            return ((ITensorActivationOperations)tensor).ApplySwish();
        }

        public static Tensor ApplyELUStatic(Tensor tensor, float alpha)
        {
            return ((ITensorActivationOperations)tensor).ApplyELU(alpha);
        }

        public static Tensor ApplySoftmaxStatic(Tensor tensor)
        {
            return ((ITensorActivationOperations)tensor).ApplySoftmax();
        }

        public static Tensor ApplyHardSigmoidStatic(Tensor tensor)
        {
            return ((ITensorActivationOperations)tensor).ApplyHardSigmoid();
        }

        public static Tensor ApplyGELUStatic(Tensor tensor)
        {
            return ((ITensorActivationOperations)tensor).ApplyGELU();
        }

        public static Tensor ApplyHardTanhStatic(Tensor tensor)
        {
            return ((ITensorActivationOperations)tensor).ApplyHardTanh();
        }

        public static Tensor ApplyMishStatic(Tensor tensor)
        {
            return ((ITensorActivationOperations)tensor).ApplyMish();
        }

        // Статические версии производных функций активации
        public static Tensor ApplySigmoidPrimeStatic(Tensor tensor)
        {
            return ((ITensorActivationOperationsPrime)tensor).ApplySigmoidPrime();
        }

        public static Tensor ApplyTanhPrimeStatic(Tensor tensor)
        {
            return ((ITensorActivationOperationsPrime)tensor).ApplyTanhPrime();
        }

        public static Tensor ApplyReLUPrimeStatic(Tensor tensor)
        {
            return ((ITensorActivationOperationsPrime)tensor).ApplyReLUPrime();
        }

        public static Tensor ApplyLeakyReLUPrimeStatic(Tensor tensor, float alpha)
        {
            return ((ITensorActivationOperationsPrime)tensor).ApplyLeakyReLUPrime(alpha);
        }

        public static Tensor ApplySoftplusPrimeStatic(Tensor tensor)
        {
            return ((ITensorActivationOperationsPrime)tensor).ApplySoftplusPrime();
        }

        public static Tensor ApplySwishPrimeStatic(Tensor tensor)
        {
            return ((ITensorActivationOperationsPrime)tensor).ApplySwishPrime();
        }

        public static Tensor ApplyGELUPrimeStatic(Tensor tensor)
        {
            return ((ITensorActivationOperationsPrime)tensor).ApplyGELUPrime();
        }

        public static Tensor ApplyHardSigmoidPrimeStatic(Tensor tensor)
        {
            return ((ITensorActivationOperationsPrime)tensor).ApplyHardSigmoidPrime();
        }

        public static Tensor ApplyHardTanhPrimeStatic(Tensor tensor)
        {
            return ((ITensorActivationOperationsPrime)tensor).ApplyHardTanhPrime();
        }

        public static Tensor ApplyMishPrimeStatic(Tensor tensor)
        {
            return ((ITensorActivationOperationsPrime)tensor).ApplyMishPrime();
        }
    }
}
