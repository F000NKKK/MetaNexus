using MetaNexus.Lib.NeuralNetwork.Tensors;
using MetaNexus.Lib.NeuralNetwork.ML.Layers;


namespace MetaNexus.Lib.NeuralNetwork.Tests.ML.Layers
{
    public class BatchNormalizationLayerTests
    {
        private BatchNormalizationLayer _batchNormalizationLayer;

        [SetUp]
        public void Setup()
        {
            // Инициализация слоя с размером входа 4 и выходом 3
            _batchNormalizationLayer = new BatchNormalizationLayer(inputSize: 4, size: 3);
        }

        [Test]
        public void TestForwardTraining()
        {
            // Создаем пример входного тензора для тренировки
            Tensor input = new Tensor(new int[] { 4, 3 });  // Входной тензор размером 4x3

            // Заполнение тензора тестовыми данными
            input[0, 0] = 1f; input[0, 1] = 2f; input[0, 2] = 3f;
            input[1, 0] = 4f; input[1, 1] = 5f; input[1, 2] = 6f;
            input[2, 0] = 7f; input[2, 1] = 8f; input[2, 2] = 9f;
            input[3, 0] = 10f; input[3, 1] = 11f; input[3, 2] = 12f;

            // Устанавливаем режим тренировки
            _batchNormalizationLayer.SetTrainingMode(true);

            // Прямой проход
            Tensor output = _batchNormalizationLayer.Forward(input);

            // Проверяем, что выходной тензор не пуст
            Assert.That(output, Is.Not.EqualTo(null));
            Assert.That(output.Shape, Is.EqualTo(input.Shape));
        }

        [Test]
        public void TestForwardInference()
        {
            // Создаем пример входного тензора для инференса
            Tensor input = new Tensor(new int[] { 4, 3 });  // Входной тензор размером 4x3

            // Заполнение тензора тестовыми данными
            input[0, 0] = 1f; input[0, 1] = 2f; input[0, 2] = 3f;
            input[1, 0] = 4f; input[1, 1] = 5f; input[1, 2] = 6f;
            input[2, 0] = 7f; input[2, 1] = 8f; input[2, 2] = 9f;
            input[3, 0] = 10f; input[3, 1] = 11f; input[3, 2] = 12f;

            // Устанавливаем режим инференса
            _batchNormalizationLayer.SetTrainingMode(false);

            // Прямой проход
            Tensor output = _batchNormalizationLayer.Forward(input);

            // Проверяем, что выходной тензор не пуст
            Assert.That(output, Is.Not.EqualTo(null));
            Assert.That(output.Shape, Is.EqualTo(input.Shape));
        }

        [Test]
        public void TestGammaBetaApplication()
        {
            // Создаем пример входного тензора для тренировки
            Tensor input = new Tensor(new int[] { 4, 3 });  // Входной тензор размером 4x3

            // Заполнение тензора тестовыми данными
            input[0, 0] = 1f; input[0, 1] = 2f; input[0, 2] = 3f;
            input[1, 0] = 4f; input[1, 1] = 5f; input[1, 2] = 6f;
            input[2, 0] = 7f; input[2, 1] = 8f; input[2, 2] = 9f;
            input[3, 0] = 10f; input[3, 1] = 11f; input[3, 2] = 12f;

            // Устанавливаем режим тренировки
            _batchNormalizationLayer.SetTrainingMode(true);

            // Прямой проход
            Tensor output = _batchNormalizationLayer.Forward(input);

            // Проверяем, что гамма и бета применяются к нормализованным данным
            Assert.That(output[0, 0], Is.Not.EqualTo(output[0, 1]));
        }
    }
}
