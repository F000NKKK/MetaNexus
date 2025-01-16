using NUnit.Framework;
using MetaNexus.Lib.NeuralNetwork.Tensors;
using System;
using MetaNexus.Lib.NeuralNetwork.Tensors.Abstractions;

namespace MetaNexus.Lib.NeuralNetwork.Tests
{
    public class TensorElementWiseOperationsTests
    {
        private ITensor _tensor1;
        private ITensor _tensor2;
        private ITensor _tensor3;

        [SetUp]
        public void Setup()
        {
            // Инициализация тестовых тензоров
            _tensor1 = new Tensor(new int[] { 2, 3 }, new float[] { 1, 2, 3, 4, 5, 6 }); // Пример с тензором 2x3
            _tensor2 = new Tensor(new int[] { 2, 3 }, new float[] { 6, 5, 4, 3, 2, 1 }); // Пример с тензором 2x3
            _tensor3 = new Tensor(new int[] { 2, 3 }, new float[] { 0, 0, 0, 0, 0, 0 }); // Пример с тензором 2x3
        }

        [Test]
        public void Test_ElementWiseAddition()
        {
            // Операция сложения
            var result = (Tensor)_tensor1.ElementWiseOperation(_tensor2, (a, b) => a + b);

            var expectedData = new float[] { 7, 7, 7, 7, 7, 7 };

            Assert.That(result.Data.Span.ToArray(), Is.EqualTo(expectedData));
        }

        [Test]
        public void Test_ElementWiseSubtraction()
        {
            // Операция вычитания
            var result = (Tensor)_tensor1.ElementWiseOperation(_tensor2, (a, b) => a - b);

            var expectedData = new float[] { -5, -3, -1, 1, 3, 5 };

            Assert.That(result.Data.Span.ToArray(), Is.EqualTo(expectedData));
        }

        [Test]
        public void Test_ElementWiseScalarMultiplication()
        {
            // Операция умножения на скаляр
            var scalar = 2f;
            var result = (Tensor)_tensor1.ElementWiseOperation(scalar, (a, b) => a * b);

            var expectedData = new float[] { 2, 4, 6, 8, 10, 12 };

            Assert.That(result.Data.Span.ToArray(), Is.EqualTo(expectedData));
        }

        [Test]
        public void Test_ElementWiseOperation_ShapeMismatch_ThrowsException()
        {
            // Проверка на несоответствие размеров тензоров
            var tensor4 = new Tensor(new int[] { 3, 2 }); // Другой размер
            Assert.Throws<InvalidOperationException>(() =>
            {
                _ = (Tensor)_tensor1.ElementWiseOperation(tensor4, (a, b) => a + b);
            });
        }

        [Test]
        public void Test_ElementWiseOperation_ScalarAddition()
        {
            // Операция сложения с скаляром
            var scalar = 3f;
            var result = (Tensor)_tensor1.ElementWiseOperation(scalar, (a, b) => a + b);

            var expectedData = new float[] { 4, 5, 6, 7, 8, 9 };

            Assert.That(result.Data.Span.ToArray(), Is.EqualTo(expectedData));
        }

        [Test]
        public void Test_ElementWiseOperation_ZeroTensor()
        {
            // Проверка операции над нулевым тензором
            var result = (Tensor)_tensor1.ElementWiseOperation(_tensor3, (a, b) => a + b);

            var expectedData = new float[] { 1, 2, 3, 4, 5, 6 };

            Assert.That(result.Data.Span.ToArray(), Is.EqualTo(expectedData));
        }
    }
}
