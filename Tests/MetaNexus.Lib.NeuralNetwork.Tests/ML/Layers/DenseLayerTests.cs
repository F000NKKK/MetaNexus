using MetaNexus.Lib.NeuralNetwork.Tensors;

namespace MetaNexus.Lib.NeuralNetwork.Tests.ML.Layers
{
    public class DenseLayerTests
    {
        private DenseLayer layer;

        [SetUp]
        public void Setup()
        {
            int inputSize = 3;
            int size = 2;

            // Инициализация весов и смещений
            var weights = new Tensor(new int[] { inputSize, size });
            var biases = new Tensor(new int[] { size });

            // Инициализация слоя с заранее заданными весами и смещениями
            layer = new DenseLayer(inputSize, size, weights, biases);
        }

        [Test]
        public void Test_Forward_ValidInput()
        {
            var inputTensor = new Tensor(new int[] { 1, 3 }); // Входной тензор с размерностью [1, 3]
            inputTensor[0, 0] = 1.0f; inputTensor[0, 1] = 2.0f; inputTensor[0, 2] = 3.0f; // Заполнение входа значениями

            var outputTensor = layer.Forward(inputTensor);

            // Проверяем, что выходной тензор не null и имеет правильную форму
            Assert.That(!Is.Equals(outputTensor, null));
            Assert.That(outputTensor.Shape.Length, Is.EqualTo(2)); // Выходной тензор должен быть двумерным
            Assert.That(outputTensor.Shape[0], Is.EqualTo(1)); // Размер первой оси [batch_size]
            Assert.That(outputTensor.Shape[1], Is.EqualTo(2)); // Размер второй оси [output_size], должно быть равно размеру слоя

            // Проводим логирование для проверки выхода
            Console.WriteLine("Выходной тензор: " + string.Join(", ", outputTensor.ToString()));
        }

        [Test]
        public void Test_Forward_ApplyActivation()
        {
            int inputSize = 3;
            int size = 2;

            // Создание тензора весов и смещений с простыми значениями
            var weights = new Tensor(new int[] { inputSize, size });
            var biases = new Tensor(new int[] { size });
            var inputTensor = new Tensor(new int[] { 1, inputSize });

            inputTensor[0, 0] = 1.0f;
            inputTensor[0, 1] = 1.0f;
            inputTensor[0, 2] = 1.0f;

            // Создание слоя
            var layer = new DenseLayer(inputSize, size, weights, biases);

            // Выполнение прямого прохода
            var outputTensor = layer.Forward(inputTensor);

            // Проверка выхода после применения активации (можно изменить на вашу активацию)
            Assert.That(!Is.Equals(outputTensor, null));
            Assert.That(outputTensor.Shape.Length, Is.EqualTo(2));
            Assert.That(outputTensor.Shape[0], Is.EqualTo(1));
            Assert.That(outputTensor.Shape[1], Is.EqualTo(size)); // Сравниваем размерность с выходом
        }

        [Test]
        public void Test_Forward_CorrectComputation()
        {
            int inputSize = 3;
            int size = 2;

            // Установка весов и смещений
            var weights = new Tensor(new int[] { inputSize, size });
            weights[0, 0] = 0.5f; weights[1, 0] = 0.5f; weights[2, 0] = 0.5f; // Вес для нейрона 0
            weights[0, 1] = 0.1f; weights[1, 1] = 0.1f; weights[2, 1] = 0.1f; // Вес для нейрона 1

            var biases = new Tensor(new int[] { size });
            biases[0] = 0.5f; // Смещение для нейрона 0
            biases[1] = 0.2f; // Смещение для нейрона 1

            // Создание слоя
            var layer = new DenseLayer(inputSize, size, weights, biases);

            // Входной тензор
            var inputTensor = new Tensor(new int[] { 1, inputSize });
            inputTensor[0, 0] = 1.0f; inputTensor[0, 1] = 1.0f; inputTensor[0, 2] = 1.0f; // Входные данные

            // Ожидаемый выход
            var expectedOutput = new Tensor(new int[] { 1, size });
            expectedOutput[0, 0] = 2.0f; // Для нейрона 0: 0.5 + 0.5 + 0.5 + 0.5 (веса + смещение)
            expectedOutput[0, 1] = 0.5f; // Для нейрона 1: 0.1 + 0.1 + 0.1 + 0.2

            var outputTensor = layer.Forward(inputTensor);

            // Сравниваем с ожидаемым выходом
            Assert.That(outputTensor[0, 0], Is.EqualTo(expectedOutput[0, 0]).Within(0.001f)); // Для нейрона 0
            Assert.That(outputTensor[0, 1], Is.EqualTo(expectedOutput[0, 1]).Within(0.001f)); // Для нейрона 1
        }
    }
}
