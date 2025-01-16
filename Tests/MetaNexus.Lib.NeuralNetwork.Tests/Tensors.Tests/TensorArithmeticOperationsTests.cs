using NUnit.Framework;
using MetaNexus.Lib.NeuralNetwork.Tensors;
using System;

namespace MetaNexus.Lib.NeuralNetwork.Tests
{
    public class TensorArithmeticOperationsTests
    {
        private Tensor _tensor1;
        private Tensor _tensor2;
        private Tensor _tensorZero;

        [SetUp]
        public void Setup()
        {
            // Инициализация тестовых данных
            _tensor1 = new Tensor(new int[] { 2, 3 }, new float[] { 1, 2, 3, 4, 5, 6 }); // Тензор 2x3
            _tensor2 = new Tensor(new int[] { 2, 3 }, new float[] { 6, 5, 4, 3, 2, 1 }); // Тензор 2x3
            _tensorZero = new Tensor(new int[] { 2, 3 }, new float[] { 0, 0, 0, 0, 0, 0 }); // Нулевой тензор 2x3
        }

        [Test]
        public void Test_Addition()
        {
            var result = _tensor1 + _tensor2;

            var expectedData = new float[] { 7, 7, 7, 7, 7, 7 };
            Assert.That(result.Data, Is.EqualTo(expectedData));
        }

        [Test]
        public void Test_Subtraction()
        {
            var result = _tensor1 - _tensor2;

            var expectedData = new float[] { -5, -3, -1, 1, 3, 5 };
            Assert.That(result.Data, Is.EqualTo(expectedData));
        }

        [Test]
        public void Test_Multiplication()
        {
            var result = _tensor1 * _tensor2;

            var expectedData = new float[] { 6, 10, 12, 12, 10, 6 };
            Assert.That(result.Data, Is.EqualTo(expectedData));
        }

        [Test]
        public void Test_Division()
        {
            var result = _tensor1 / _tensor2;

            var expectedData = new float[] { 1f / 6, 2f / 5, 3f / 4, 4f / 3, 5f / 2, 6f / 1 };
            Assert.That(result.Data, Is.EqualTo(expectedData).Within(0.0001f));
        }

        [Test]
        public void Test_ScalarAddition()
        {
            var scalar = 3f;
            var result = _tensor1 + scalar;

            var expectedData = new float[] { 4, 5, 6, 7, 8, 9 };
            Assert.That(result.Data, Is.EqualTo(expectedData));
        }

        [Test]
        public void Test_ScalarSubtraction()
        {
            var scalar = 2f;
            var result = _tensor1 - scalar;

            var expectedData = new float[] { -1, 0, 1, 2, 3, 4 };
            Assert.That(result.Data, Is.EqualTo(expectedData));
        }

        [Test]
        public void Test_ScalarMultiplication()
        {
            var scalar = 2f;
            var result = _tensor1 * scalar;

            var expectedData = new float[] { 2, 4, 6, 8, 10, 12 };
            Assert.That(result.Data, Is.EqualTo(expectedData));
        }

        [Test]
        public void Test_ScalarDivision()
        {
            var scalar = 2f;
            var result = _tensor1 / scalar;

            var expectedData = new float[] { 0.5f, 1, 1.5f, 2, 2.5f, 3 };
            Assert.That(result.Data, Is.EqualTo(expectedData).Within(0.0001f));
        }

        [Test]
        public void Test_ZeroDivision_ThrowsException()
        {
            Assert.Throws<DivideByZeroException>(() =>
            {
                _ = _tensor1 / 0f;
            });
        }

        [Test]
        public void Test_ZeroTensorDivision_ThrowsException()
        {
            Assert.Throws<DivideByZeroException>(() =>
            {
                _ = _tensor1 / _tensorZero;
            });
        }

        [Test]
        public void Test_ShapeMismatch_ThrowsException()
        {
            var tensorMismatch = new Tensor(new int[] { 3, 2 }, new float[] { 1, 2, 3, 4, 5, 6 }); // Тензор 3x2

            Assert.Throws<InvalidOperationException>(() =>
            {
                _ = _tensor1 + tensorMismatch;
            });
        }
    }
}
