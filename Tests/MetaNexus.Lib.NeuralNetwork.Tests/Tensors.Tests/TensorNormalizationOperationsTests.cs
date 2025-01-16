using MetaNexus.Lib.NeuralNetwork.Tensors;
using NUnit.Framework;

namespace MetaNexus.Lib.NeuralNetwork.Tests
{
    public class TensorNormalizationOperationsTests
    {
        [SetUp]
        public void Setup()
        {
        }

        [Test]
        public void Test_Normalize()
        {
            var tensor = new Tensor(new int[] { 2, 2 }, new float[] { 1.0f, 2.0f, 3.0f, 4.0f });
            var normalizedTensor = tensor.Normalize();

            // Ожидаемый результат: среднее = 2.5, стандартное отклонение = 1.118, нормализованные значения
            var expected = new float[] { -1.3416406f, -0.4472136f, 0.4472136f, 1.3416406f };

            Assert.That(normalizedTensor.Data.Span.ToArray(), Is.EqualTo(expected).Within(0.0001f));
        }

        [Test]
        public void Test_BatchNormalize()
        {
            var tensor = new Tensor(new int[] { 2, 2 }, new float[] { 1.0f, 2.0f, 3.0f, 4.0f });
            var mean = new Tensor(new int[] { 2, 2 }, new float[] { 2.0f, 2.0f, 2.0f, 2.0f });
            var variance = new Tensor(new int[] { 2, 2 }, new float[] { 1.0f, 1.0f, 1.0f, 1.0f });

            var normalizedTensor = tensor.BatchNormalize(mean, variance);

            // Ожидаемый результат: нормализованные значения
            var expected = new float[] { -1.0f, 0.0f, 1.0f, 2.0f };

            Assert.That(normalizedTensor.Data.Span.ToArray(), Is.EqualTo(expected).Within(0.0001f));
        }

        [Test]
        public void Test_MinMaxNormalize()
        {
            var tensor = new Tensor(new int[] { 2, 2 }, new float[] { 1.0f, 2.0f, 3.0f, 4.0f });
            var normalizedTensor = tensor.MinMaxNormalize();

            // Ожидаемый результат: минимальные значения становятся 0, максимальные 1
            var expected = new float[] { 0.0f, 0.33333334f, 0.6666667f, 1.0f };

            Assert.That(normalizedTensor.Data.Span.ToArray(), Is.EqualTo(expected).Within(0.0001f));
        }

        [Test]
        public void Test_ChannelNormalize()
        {
            var tensor = new Tensor(new int[] { 2, 2 }, new float[] { 1.0f, 2.0f, 3.0f, 4.0f });
            var mean = new Tensor(new int[] { 2, 2 }, new float[] { 2.0f, 2.0f, 2.0f, 2.0f });
            var variance = new Tensor(new int[] { 2, 2 }, new float[] { 1.0f, 1.0f, 1.0f, 1.0f });

            var normalizedTensor = tensor.ChannelNormalize(mean, variance);

            // Ожидаемый результат: нормализованные значения
            var expected = new float[] { -1.0f, 0.0f, 1.0f, 2.0f };

            Assert.That(normalizedTensor.Data.Span.ToArray(), Is.EqualTo(expected).Within(0.0001f));
        }

        [Test]
        public void Test_Standardize()
        {
            var tensor = new Tensor(new int[] { 2, 2 }, new float[] { 1.0f, 2.0f, 3.0f, 4.0f });
            var standardizedTensor = tensor.Standardize();

            // Ожидаемый результат: среднее = 2.5, стандартное отклонение = 1.118, стандартизированные значения
            var expected = new float[] { -1.3416406f, -0.4472136f, 0.4472136f, 1.3416406f };

            Assert.That(standardizedTensor.Data.Span.ToArray(), Is.EqualTo(expected).Within(0.0001f));
        }

        [Test]
        public void Test_LabelNormalize()
        {
            var tensor = new Tensor(new int[] { 2, 2 }, new float[] { 2.0f, 4.0f, 6.0f, 8.0f });
            var numClasses = 10;

            var normalizedTensor = tensor.LabelNormalize(numClasses);

            // Ожидаемый результат: каждое значение делится на количество классов
            var expected = new float[] { 0.2f, 0.4f, 0.6f, 0.8f };

            Assert.That(normalizedTensor.Data.Span.ToArray(), Is.EqualTo(expected).Within(0.0001f));
        }
    }
}
