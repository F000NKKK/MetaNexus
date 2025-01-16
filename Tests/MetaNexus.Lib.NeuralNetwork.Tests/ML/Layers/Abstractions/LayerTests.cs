using MetaNexus.Lib.NeuralNetwork.ML.Layers.Abstractions;
using MetaNexus.Lib.NeuralNetwork.Tensors;

namespace MetaNexus.Lib.NeuralNetwork.Tests.ML.Layers.Abstractions
{
    public class LayerTests
    {
        [SetUp]
        public void Setup()
        {
        }

        [Test]
        public void Test_Layer_Constructor_ValidArguments()
        {
            int inputSize = 3;
            int size = 2;

            var layer = new MockLayer(inputSize, size);

            Assert.That(layer.Size, Is.EqualTo(size));
            Assert.That(layer.InputSize, Is.EqualTo(inputSize));
            Assert.That(layer.GetWeights().Shape, Is.EqualTo(new int[] { inputSize, size }));
            Assert.That(layer.GetBiases().Shape, Is.EqualTo(new int[] { size }));
        }

        [Test]
        public void Test_Layer_Constructor_InvalidInputSize()
        {
            Assert.Throws<ArgumentException>(() => new MockLayer(-1, 2));
        }

        [Test]
        public void Test_Layer_Constructor_InvalidSize()
        {
            Assert.Throws<ArgumentException>(() => new MockLayer(3, -2));
        }

        [Test]
        public void Test_Layer_SetWeights_Valid()
        {
            int inputSize = 3;
            int size = 2;
            var layer = new MockLayer(inputSize, size);

            var newWeights = new Tensor(new int[] { inputSize, size });
            layer.SetWeights(newWeights);

            Assert.That(layer.GetWeights(), Is.EqualTo(newWeights));
        }

        [Test]
        public void Test_Layer_SetWeights_Invalid()
        {
            int inputSize = 3;
            int size = 2;
            var layer = new MockLayer(inputSize, size);

            var invalidWeights = new Tensor(new int[] { inputSize, size + 1 });
            Assert.Throws<ArgumentException>(() => layer.SetWeights(invalidWeights));
        }

        [Test]
        public void Test_Layer_SetBiases_Valid()
        {
            int inputSize = 3;
            int size = 2;
            var layer = new MockLayer(inputSize, size);

            var newBiases = new Tensor(new int[] { size });
            layer.SetBiases(newBiases);

            Assert.That(layer.GetBiases(), Is.EqualTo(newBiases));
        }

        [Test]
        public void Test_Layer_SetBiases_Invalid()
        {
            int inputSize = 3;
            int size = 2;
            var layer = new MockLayer(inputSize, size);

            var invalidBiases = new Tensor(new int[] { size + 1 });
            Assert.Throws<ArgumentException>(() => layer.SetBiases(invalidBiases));
        }

        [Test]
        public void Test_Forward_Pass()
        {
            int inputSize = 3;
            int size = 2;
            var layer = new MockLayer(inputSize, size);

            var inputTensor = new Tensor(new int[] { inputSize });
            var outputTensor = layer.Forward(inputTensor);

            Assert.That(outputTensor.Shape[1], Is.EqualTo(size));
        }


        [Test]
        public void Test_Backward_Valid()
        {
            int inputSize = 3;
            int size = 2;
            var layer = new MockLayer(inputSize, size);

            var input = new Tensor(new int[] { 1, inputSize }); // Двумерный тензор
            layer.Forward(input);

            var delta = new Tensor(new int[] { size });
            var learningRate = 0.01f;

            var previousDelta = layer.Backward(delta, learningRate);

            Assert.That(previousDelta.Shape[1], Is.EqualTo(inputSize));
        }

        // Мок-реализация слоя для тестирования
        private class MockLayer : Layer
        {
            public MockLayer(int inputSize, int size)
                : base(inputSize, size)
            {
            }

            public MockLayer(int inputSize, int size, Tensor weights, Tensor biases)
                : base(inputSize, size, weights, biases)
            {
            }

            public override Tensor Forward(Tensor input)
            {
                if (input.Shape.Length == 1)
                {
                    input = input.Reshape(new int[] { 1, input.Shape[0] }); // Преобразуем в двумерный тензор
                }

                this.input = input; // Сохраняем входной тензор для использования в Backward

                // Умножаем вход на веса (результат [1, outputSize])
                Tensor output = input.Dot(weights);

                // Проверяем, что смещения имеют правильную форму
                if (biases.Shape[0] == 1)
                {
                    // Если смещения одномерные, их можно просто прибавить
                    output += biases; // [1, outputSize]
                }
                else
                {
                    // В случае, если смещения имеют форму [outputSize], то нужно выполнить расширение размерности
                    output += biases.Reshape(new int[] { 1, biases.Shape[0] });
                }

                return output;
            }

        }
    }
}
