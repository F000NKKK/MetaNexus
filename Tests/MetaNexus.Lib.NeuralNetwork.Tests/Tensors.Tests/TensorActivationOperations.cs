using MetaNexus.Lib.NeuralNetwork.Tensors;
using MetaNexus.Lib.NeuralNetwork.Tensors.Abstractions;
using NUnit.Framework;
using System;
using System.Linq;

namespace MetaNexus.Lib.NeuralNetwork.Tests
{
    public class TensorActivationOperations
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
            var result = tensor.ApplyReLU();
            Assert.That(result.FlattenFloatArray(), Is.EqualTo(new float[] { 0f, 0f, 1f, 2f, 0f, 3f }), "ReLU test failed.");
        }

        [Test]
        public void ApplyLeakyReLU_Test()
        {
            float alpha = 0.1f;
            var result = tensor.ApplyLeakyReLU(alpha);
            var expected = ComputeExpectedLeakyReLU(tensor.FlattenFloatArray(), alpha);
            Assert.That(result.FlattenFloatArray(), Is.EqualTo(expected), "LeakyReLU test failed.");
        }

        [Test]
        public void ApplySigmoid_Test()
        {
            var result = tensor.ApplySigmoid();
            var expected = ComputeExpectedSigmoid(tensor.FlattenFloatArray());
            Assert.That(result.FlattenFloatArray(), Is.EqualTo(expected).Within(1e-6f), "Sigmoid test failed.");
        }

        [Test]
        public void ApplyTanh_Test()
        {
            var result = tensor.ApplyTanh();
            var expected = tensor.FlattenFloatArray().Select(MathF.Tanh).ToArray();
            Assert.That(result.FlattenFloatArray(), Is.EqualTo(expected).Within(1e-6f), "Tanh test failed.");
        }

        [Test]
        public void ApplySoftmax_Test()
        {
            var result = tensor.ApplySoftmax();
            var expected = ComputeExpectedSoftmax(tensor.FlattenFloatArray());
            Assert.That(result.FlattenFloatArray(), Is.EqualTo(expected).Within(1e-6f), "Softmax test failed.");
        }

        [Test]
        public void ApplySwish_Test()
        {
            var result = tensor.ApplySwish();
            var expected = tensor.FlattenFloatArray().Select(x => x / (1f + MathF.Exp(-x))).ToArray();
            Assert.That(result.FlattenFloatArray(), Is.EqualTo(expected).Within(1e-6f), "Swish test failed.");
        }

        [Test]
        public void ApplyMish_Test()
        {
            var result = tensor.ApplyMish();
            var expected = tensor.FlattenFloatArray().Select(x => x * MathF.Tanh(MathF.Log(1f + MathF.Exp(x)))).ToArray();
            Assert.That(result.FlattenFloatArray(), Is.EqualTo(expected).Within(1e-6f), "Mish test failed.");
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
