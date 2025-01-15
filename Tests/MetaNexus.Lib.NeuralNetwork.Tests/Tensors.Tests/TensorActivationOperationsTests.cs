using MetaNexus.Lib.NeuralNetwork.Tensors;
using MetaNexus.Lib.NeuralNetwork.Tensors.Abstractions;

namespace MetaNexus.Lib.NeuralNetwork.Tests
{
    public class TensorActivationOperationsTests
    {
        private ITensor tensor;

        [SetUp]
        public void Setup()
        {
            tensor = new Tensor(new int[] { 2, 3 }, new float[] { -1f, 0f, 1f, 2f, -2f, 3f });
        }

        [Test]
        public void ApplyReLU_Test()
        {
            var result = Tensor.ApplyReLUStatic(tensor);
            Assert.That(result.FlattenFloatArray(), Is.EqualTo(new float[] { 0f, 0f, 1f, 2f, 0f, 3f }), "ReLU test failed.");
        }

        [Test]
        public void ApplyLeakyReLU_Test()
        {
            float alpha = 0.1f;
            var result = Tensor.ApplyLeakyReLUStatic(tensor, alpha);
            var expected = ComputeExpectedLeakyReLU(tensor.FlattenFloatArray(), alpha);
            Assert.That(result.FlattenFloatArray(), Is.EqualTo(expected), "LeakyReLU test failed.");
        }

        [Test]
        public void ApplySigmoid_Test()
        {
            var result = Tensor.ApplySigmoidStatic(tensor);
            var expected = ComputeExpectedSigmoid(tensor.FlattenFloatArray());
            Assert.That(result.FlattenFloatArray(), Is.EqualTo(expected).Within(1e-6f), "Sigmoid test failed.");
        }

        [Test]
        public void ApplyTanh_Test()
        {
            var result = Tensor.ApplyTanhStatic(tensor);
            var expected = tensor.FlattenFloatArray().Select(MathF.Tanh).ToArray();
            Assert.That(result.FlattenFloatArray(), Is.EqualTo(expected).Within(1e-6f), "Tanh test failed.");
        }

        [Test]
        public void ApplySoftmax_Test()
        {
            var result = Tensor.ApplySoftmaxStatic(tensor);
            var expected = ComputeExpectedSoftmax(tensor.FlattenFloatArray());
            Assert.That(result.FlattenFloatArray(), Is.EqualTo(expected).Within(1e-6f), "Softmax test failed.");
        }

        [Test]
        public void ApplySwish_Test()
        {
            var result = Tensor.ApplySwishStatic(tensor);
            var expected = tensor.FlattenFloatArray().Select(x => x / (1f + MathF.Exp(-x))).ToArray();
            Assert.That(result.FlattenFloatArray(), Is.EqualTo(expected).Within(1e-6f), "Swish test failed.");
        }

        [Test]
        public void ApplyMish_Test()
        {
            var result = Tensor.ApplyMishStatic(tensor);
            var expected = tensor.FlattenFloatArray().Select(x => x * MathF.Tanh(MathF.Log(1f + MathF.Exp(x)))).ToArray();
            Assert.That(result.FlattenFloatArray(), Is.EqualTo(expected).Within(1e-6f), "Mish test failed.");
        }

        [Test]
        public void ApplyELU_Test()
        {
            float alpha = 1.0f;
            var result = Tensor.ApplyELUStatic(tensor, alpha);
            Assert.That(result.FlattenFloatArray(), Is.EqualTo(new float[]
            {
                alpha * (MathF.Exp(-1f) - 1f),
                0f,
                1f,
                2f,
                alpha * (MathF.Exp(-2f) - 1f),
                3f
            }).Within(1e-6f));
        }

        [Test]
        public void ApplySoftplus_Test()
        {
            var result = Tensor.ApplySoftplusStatic(tensor);
            Assert.That(result.FlattenFloatArray(), Is.EqualTo(new float[]
            {
                MathF.Log(1f + MathF.Exp(-1f)),
                MathF.Log(2f),
                MathF.Log(1f + MathF.Exp(1f)),
                MathF.Log(1f + MathF.Exp(2f)),
                MathF.Log(1f + MathF.Exp(-2f)),
                MathF.Log(1f + MathF.Exp(3f))
            }).Within(1e-6f));
        }

        [Test]
        public void ApplyHardSigmoid_Test()
        {
            var result = Tensor.ApplyHardSigmoidStatic(tensor);
            Assert.That(result.FlattenFloatArray(), Is.EqualTo(new float[]
            {
                MathF.Min(MathF.Max((-1f + 1f) / 2f, 0f), 1f),
                MathF.Min(MathF.Max((0f + 1f) / 2f, 0f), 1f),
                MathF.Min(MathF.Max((1f + 1f) / 2f, 0f), 1f),
                MathF.Min(MathF.Max((2f + 1f) / 2f, 0f), 1f),
                MathF.Min(MathF.Max((-2f + 1f) / 2f, 0f), 1f),
                MathF.Min(MathF.Max((3f + 1f) / 2f, 0f), 1f)
            }).Within(1e-6f));
        }

        [Test]
        public void ApplyGELU_Test()
        {
            var result = Tensor.ApplyGELUStatic(tensor);
            Assert.That(result.FlattenFloatArray(), Is.EqualTo(new float[]
            {
                -1f * (1f + MathF.Tanh(MathF.Sqrt(2f / MathF.PI) * (-1f + 0.5f))) / 2f,
                0f,
                1f * (1f + MathF.Tanh(MathF.Sqrt(2f / MathF.PI) * (1f + 0.5f))) / 2f,
                2f * (1f + MathF.Tanh(MathF.Sqrt(2f / MathF.PI) * (2f + 0.5f))) / 2f,
                -2f * (1f + MathF.Tanh(MathF.Sqrt(2f / MathF.PI) * (-2f + 0.5f))) / 2f,
                3f * (1f + MathF.Tanh(MathF.Sqrt(2f / MathF.PI) * (3f + 0.5f))) / 2f
            }).Within(1e-6f));
        }

        [Test]
        public void ApplyHardTanh_Test()
        {
            var result = Tensor.ApplyHardTanhStatic(tensor);
            Assert.That(result.FlattenFloatArray(), Is.EqualTo(new float[]
            {
                MathF.Max(MathF.Min(-1f, 1f), -1f),
                MathF.Max(MathF.Min(0f, 1f), -1f),
                MathF.Max(MathF.Min(1f, 1f), -1f),
                MathF.Max(MathF.Min(2f, 1f), -1f),
                MathF.Max(MathF.Min(-2f, 1f), -1f),
                MathF.Max(MathF.Min(3f, 1f), -1f)
            }).Within(1e-4f));
        }

        [Test]
        public void ApplySoftSign_Test()
        {
            var result = Tensor.ApplySoftSignStatic(tensor);
            Assert.That(result.FlattenFloatArray(), Is.EqualTo(new float[]
            {
                -1f / (1f + MathF.Abs(-1f)),
                0f / (1f + MathF.Abs(0f)),
                1f / (1f + MathF.Abs(1f)),
                2f / (1f + MathF.Abs(2f)),
                -2f / (1f + MathF.Abs(-2f)),
                3f / (1f + MathF.Abs(3f))
            }).Within(1e-6f));
        }

        [Test]
        public void ApplySigmoidPrime_Test()
        {
            var result = Tensor.ApplySigmoidPrimeStatic(tensor);
            Assert.That(result.Data, Is.EqualTo(new float[]
            {
                0.19661193f,
                0.25f,
                0.19661193f,
                0.10499359f,
                0.10499359f,
                0.04517666f
            }).Within(1e-6f));
        }

        [Test]
        public void ApplyTanhPrime_Test()
        {
            var result = Tensor.ApplyTanhPrimeStatic(tensor);
            Assert.That(result.Data, Is.EqualTo(new float[]
            {
                0.41997434f,
                1f,
                0.41997434f,
                0.07065082f,
                0.07065082f,
                0.00986603f
            }).Within(1e-6f));
        }

        [Test]
        public void ApplyReLUPrime_Test()
        {
            var result = Tensor.ApplyReLUPrimeStatic(tensor);
            Assert.That(result.Data, Is.EqualTo(new float[] { 0f, 0f, 1f, 1f, 0f, 1f }));
        }

        [Test]
        public void ApplyLeakyReLUPrime_Test()
        {
            float alpha = 0.1f;
            var result = Tensor.ApplyLeakyReLUPrimeStatic(tensor, alpha);
            Assert.That(result.Data, Is.EqualTo(new float[] { 0.1f, 0.1f, 1f, 1f, 0.1f, 1f }));
        }

        [Test]
        public void ApplySoftplusPrime_Test()
        {
            var result = Tensor.ApplySoftplusPrimeStatic(tensor);
            Assert.That(result.Data, Is.EqualTo(new float[]
            {
                0.26894142f,
                0.5f,
                0.73105858f,
                0.88079708f,
                0.11920292f,
                0.95257413f
            }).Within(1e-6f));
        }

        [Test]
        public void ApplySwishPrime_Test()
        {
            var result = Tensor.ApplySwishPrimeStatic(tensor);
            Assert.That(result.Data, Is.EqualTo(new float[]
            {
                0.07232949f,
                0.5f,
                0.92767051f,
                1.09078431f,
                -0.09078424f,
                1.088103f
            }).Within(1e-5f));
        }



        [Test]
        public void ApplyGELUPrime_Test()
        {
            var result = Tensor.ApplyGELUPrimeStatic(tensor);
            Assert.That(result.Data, Is.EqualTo(new float[]
            {
                0.130211473f, 
                0.5f,
                0.869788527f,
                0.986797214f,
                0.0132027464f,
                0.999565184f
            }
            ).Within(1e-6f));
        }

        [Test]
        public void ApplyHardSigmoidPrime_Test()
        {
            var result = Tensor.ApplyHardSigmoidPrimeStatic(tensor);
            Assert.That(result.Data, Is.EqualTo(new float[] { 0.2f, 0.2f, 0.2f, 0.2f, 0.2f, 0f }));
        }

        [Test]
        public void ApplyHardTanhPrime_Test()
        {
            var result = Tensor.ApplyHardTanhPrimeStatic(tensor);
            Assert.That(result.Data, Is.EqualTo(new float[] { 0f, 1f, 0f, 0f, 0f, 0f }));
        }

        [Test]
        public void ApplyMishPrime_Test()
        {
            var result = Tensor.ApplyMishPrimeStatic(tensor);
            Assert.That(result.Data, Is.EqualTo(new float[]
            {
                0.30058962f,
                0.65f,
                0.80693483f,
                0.89599138f,
                0.03514066f,
                0.95487081f
            }).Within(1e-6f));
        }

        [Test]
        public void ApplyIdentity_Test()
        {
            var result = Tensor.ApplyIdentityStatic(tensor);
            Assert.That(result.Data, Is.EqualTo(new float[] { -1f, 0f, 1f, 2f, -2f, 3f }));
        }

        private float[] ComputeExpectedLeakyReLU(float[] input, float alpha)
        {
            return input.Select(x => x > 0 ? x : x * alpha).ToArray();
        }

        private float[] ComputeExpectedSigmoid(float[] input)
        {
            return input.Select(x => 1f / (1f + MathF.Exp(-x))).ToArray();
        }

        private float[] ComputeExpectedSoftmax(float[] input)
        {
            var expValues = input.Select(MathF.Exp).ToArray();
            var sum = expValues.Sum();
            return expValues.Select(x => x / sum).ToArray();
        }
    }
}
