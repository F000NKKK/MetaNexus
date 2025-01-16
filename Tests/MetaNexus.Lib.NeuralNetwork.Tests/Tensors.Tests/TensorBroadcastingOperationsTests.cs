using NUnit.Framework;
using MetaNexus.Lib.NeuralNetwork.Tensors;
using System;

namespace MetaNexus.Lib.NeuralNetwork.Tests
{
    public class TensorBroadcastingOperationsTests
    {
        private Tensor _tensor1;
        private Tensor _tensor2;
        private Tensor _tensor3;

        [SetUp]
        public void Setup()
        {
            // Инициализация тензоров для тестов
            _tensor1 = new Tensor(new int[] { 2, 3 }, new float[] { 1, 2, 3, 4, 5, 6 });
            _tensor2 = new Tensor(new int[] { 1, 3 }, new float[] { 1, 2, 3 });
            _tensor3 = new Tensor(new int[] { 2, 1 }, new float[] { 1, 2 });
        }

        [Test]
        public void Test_CanBroadcast_ValidShapes_ReturnsTrue()
        {
            // Тензоры, которые должны правильно трансляцироваться
            Assert.That(_tensor1.CanBroadcast(_tensor2), Is.True);  // (2,3) и (1,3)
            Assert.That(_tensor1.CanBroadcast(_tensor3), Is.True);  // (2,3) и (2,1)
            Assert.That(_tensor2.CanBroadcast(_tensor3), Is.True);  // (1,3) и (2,1)
        }

        [Test]
        public void Test_CanBroadcast_InvalidShapes_ReturnsFalse()
        {
            // Тензоры с несовместимыми формами
            var invalidTensor = new Tensor(new int[] { 3, 3 }, new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 });
            Assert.That(_tensor1.CanBroadcast(invalidTensor), Is.False);  // (2,3) и (3,3)
        }

        [Test]
        public void Test_BroadcastAdd_ValidBroadcast_ReturnsCorrectResult()
        {
            var tensor1 = new Tensor(new int[] { 2, 3 }, new float[] { 1, 2, 3, 4, 5, 6 });
            var tensor2 = new Tensor(new int[] { 1, 3 }, new float[] { 1, 2, 3 });

            var result = tensor1.BroadcastAdd(tensor2);
            var expectedData = new float[] { 2, 4, 6, 5, 7, 9 }; // (2,3) + (1,3) -> результат (2,3)

            Assert.That(result.Data, Is.EqualTo(expectedData));
        }

        [Test]
        public void Test_BroadcastSubtract_ValidBroadcast_ReturnsCorrectResult()
        {
            var result = _tensor1.BroadcastSubtract(_tensor3);
            var expectedData = new float[] { 0, 1, 2, 2, 3, 4 }; // (2,3) - (2,1) -> результат (2,3)

            Assert.That(result.Data, Is.EqualTo(expectedData));
        }

        [Test]
        public void Test_BroadcastMultiply_ValidBroadcast_ReturnsCorrectResult()
        {
            var result = _tensor1.BroadcastMultiply(_tensor2);
            var expectedData = new float[] { 1, 4, 9, 4, 10, 18 }; // (2,3) * (1,3) -> результат (2,3)

            Assert.That(result.Data, Is.EqualTo(expectedData));
        }

        [Test]
        public void Test_BroadcastDivide_ValidBroadcast_ReturnsCorrectResult()
        {
            var result = _tensor1.BroadcastDivide(_tensor3);
            var expectedData = new float[] { 1, 2, 3, 2, 2.5f, 3 }; // (2,3) / (2,1) -> результат (2,3)

            Assert.That(result.Data, Is.EqualTo(expectedData));
        }

        [Test]
        public void Test_BroadcastAdd_InvalidBroadcast_ThrowsException()
        {
            var invalidTensor = new Tensor(new int[] { 3, 3 }, new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 });
            // Попытка сложения несовместимых тензоров
            Assert.Throws<InvalidOperationException>(() => _tensor1.BroadcastAdd(invalidTensor));
        }

        [Test]
        public void Test_BroadcastDivide_ZeroTensor_ThrowsException()
        {
            var zeroTensor = new Tensor(new int[] { 2, 3 }, new float[] { 0, 0, 0, 0, 0, 0 });
            // Проверка деления на ноль
            Assert.Throws<DivideByZeroException>(() => _tensor1.BroadcastDivide(zeroTensor));
        }

        [Test]
        public void Test_BroadcastAdd_ExpandDimensions_ReturnsCorrectResult()
        {
            var tensor1 = new Tensor(new int[] { 1, 3 }, new float[] { 1, 2, 3 });
            var tensor2 = new Tensor(new int[] { 2, 3 }, new float[] { 4, 5, 6, 7, 8, 9 });
            var result = tensor2.BroadcastAdd(tensor1);

            var expectedData = new float[] { 5, 7, 9, 8, 10, 12 };
            Assert.That(result.Data, Is.EqualTo(expectedData));
        }
    }
}
