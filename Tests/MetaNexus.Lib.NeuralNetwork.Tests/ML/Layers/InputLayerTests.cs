using MetaNexus.Lib.NeuralNetwork.Tensors;
using static MetaNexus.Lib.NeuralNetwork.ML.Layers.Abstractions.Layer;

namespace MetaNexus.Lib.NeuralNetwork.Tests.ML.Layers
{
    public class InputLayerTests
    {
        private InputLayer layer;

        [SetUp]
        public void Setup()
        {
            // Создаем слой с заданными параметрами
            int inputSize = 3;
            int size = 2;
            var activationFunction = (ActivationFunc)((x) => Tensor.ApplyReLUStatic(x)); // Просто пример функции активации
            var activationPrimeFunction = (ActivationPrimeFunc)((x) => Tensor.ApplyReLUPrimeStatic(x)); // Пример производной функции

            layer = new InputLayer(inputSize, size, activationFunction, activationPrimeFunction);
        }

        [Test]
        public void Test_Forward_Pass()
        {
            int inputSize = 3;
            var inputTensor = new Tensor(new int[] { 1, inputSize }); // Создаем входной тензор с размерностью [1, inputSize]
            var outputTensor = layer.Forward(inputTensor);

            // Проверяем, что выходной тензор имеет ту же форму, что и входной
            Assert.That(outputTensor.Shape, Is.EqualTo(inputTensor.Shape));
        }

        [Test]
        public void Test_Forward_ValidInput()
        {
            int inputSize = 5;
            var inputTensor = new Tensor(new int[] { 1, inputSize }); // Тензор с размерностью [1, 5]
            var outputTensor = layer.Forward(inputTensor);

            // Убедимся, что выходной тензор не null и имеет правильную форму
            Assert.That(!Is.Equals(outputTensor, null));
            Assert.That(outputTensor.Shape.Length, Is.EqualTo(2)); // Должен быть двумерный тензор
            Assert.That(outputTensor.Shape[1], Is.EqualTo(inputSize)); // Размер второй оси должен быть inputSize
        }

        [Test]
        public void Test_Forward_EmptyInput()
        {
            // Пустой тензор, можно проверить поведение
            var inputTensor = new Tensor(new int[] { 0 }); // Создаем тензор с размерностью [0]
            var outputTensor = layer.Forward(inputTensor);

            // Ожидаем, что результат будет пустым тензором
            Assert.That(outputTensor.Shape.Length, Is.EqualTo(1));
            Assert.That(outputTensor.Shape[0], Is.EqualTo(0));
        }
    }
}
