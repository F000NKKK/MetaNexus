using NUnit.Framework;
using MetaNexus.Lib.NeuralNetwork.Tensors;

namespace MetaNexus.Lib.NeuralNetwork.Tests
{
    [TestFixture]
    public class TensorAggregationOperationsTests
    {
        private Tensor _tensor;

        [SetUp]
        public void Setup()
        {
            // Пример для тензора 2x2
            _tensor = new Tensor(new[] { 2, 2 })
            {
                [0, 0] = 1f,
                [0, 1] = 3f,
                [1, 0] = 2f,
                [1, 1] = 4f
            };
        }

        [Test]
        public void TestMax()
        {
            Assert.That(_tensor.Max(), Is.EqualTo(4f));
        }

        [Test]
        public void TestMin()
        {
            Assert.That(_tensor.Min(), Is.EqualTo(1f));
        }

        [Test]
        public void TestAvg()
        {
            Assert.That(_tensor.Avg(), Is.EqualTo(2.5f).Within(0.0001f));
        }

        [Test]
        public void TestSum()
        {
            Assert.That(_tensor.Sum(), Is.EqualTo(10f));
        }

        [Test]
        public void TestSum_AggregateByAxis_0()
        {
            var sumAxis0 = _tensor.Sum(0);  // Суммируем по первой оси
            Assert.That(sumAxis0[0, 0], Is.EqualTo(3f));
            Assert.That(sumAxis0[0, 1], Is.EqualTo(7f));
        }

        [Test]
        public void TestSum_AggregateByAxis_1()
        {
            var sumAxis1 = _tensor.Sum(1);  // Суммируем по второй оси
            Assert.That(sumAxis1[0, 0], Is.EqualTo(4f));
            Assert.That(sumAxis1[1, 0], Is.EqualTo(6f));
        }

        [Test]
        public void TestEmptyTensor_Max()
        {
            var emptyTensor = new Tensor(new[] { 0 });
            Assert.That(() => emptyTensor.Max(), Throws.InvalidOperationException.With.Message.EqualTo("Тензор пуст"));
        }

        [Test]
        public void TestEmptyTensor_Min()
        {
            var emptyTensor = new Tensor(new[] { 0 });
            Assert.That(() => emptyTensor.Min(), Throws.InvalidOperationException.With.Message.EqualTo("Тензор пуст"));
        }

        [Test]
        public void TestEmptyTensor_Avg()
        {
            var emptyTensor = new Tensor(new[] { 0 });
            Assert.That(() => emptyTensor.Avg(), Throws.InvalidOperationException.With.Message.EqualTo("Тензор пуст"));
        }

        [Test]
        public void TestEmptyTensor_Sum()
        {
            var emptyTensor = new Tensor(new[] { 0 });
            Assert.That(() => emptyTensor.Sum(), Throws.InvalidOperationException.With.Message.EqualTo("Тензор пуст"));
        }

        [Test]
        public void TestInvalidAxis_ForSum()
        {
            var tensor = new Tensor(new[] { 2, 2 });
            Assert.That(() => tensor.Sum(2), Throws.TypeOf<ArgumentOutOfRangeException>());
        }

        [Test]
        public void TestSingleElementTensor_Max()
        {
            var singleElementTensor = new Tensor(new[] { 1, 1 }) { [0, 0] = 42f };
            Assert.That(singleElementTensor.Max(), Is.EqualTo(42f));
        }

        [Test]
        public void TestSingleElementTensor_Min()
        {
            var singleElementTensor = new Tensor(new[] { 1, 1 }) { [0, 0] = 42f };
            Assert.That(singleElementTensor.Min(), Is.EqualTo(42f));
        }

        [Test]
        public void TestAvg_WithNegativeNumbers()
        {
            var tensor = new Tensor(new[] { 2, 2 })
            {
                [0, 0] = -1f,
                [0, 1] = -3f,
                [1, 0] = -2f,
                [1, 1] = -4f
            };
            Assert.That(tensor.Avg(), Is.EqualTo(-2.5f).Within(0.0001f));
        }

        [Test]
        public void TestSum_WithNegativeNumbers()
        {
            var tensor = new Tensor(new[] { 2, 2 })
            {
                [0, 0] = -1f,
                [0, 1] = -3f,
                [1, 0] = -2f,
                [1, 1] = -4f
            };
            Assert.That(tensor.Sum(), Is.EqualTo(-10f));
        }
    }
}
