using MetaNexus.Lib.NeuralNetwork.Tensors;

namespace MetaNexus.Lib.NeuralNetwork.Tests.Tensors
{

    public class TensorShapeOperationsTests
    {
        [SetUp]
        public void Setup()
        {
        }

        [Test]
        public void Transpose_With2DArray_ReturnsTransposedTensor()
        {
            // Arrange
            int[] shape = { 2, 3 };
            float[] data = { 1f, 2f, 3f, 4f, 5f, 6f };
            var tensor = new Tensor(shape, data);

            // Act
            var transposedTensor = tensor.Transpose();

            // Assert
            Assert.That(transposedTensor.Shape[0], Is.EqualTo(3));
            Assert.That(transposedTensor.Shape[1], Is.EqualTo(2));
            Assert.That(transposedTensor[0, 0], Is.EqualTo(1f));
            Assert.That(transposedTensor[0, 1], Is.EqualTo(4f));
            Assert.That(transposedTensor[1, 0], Is.EqualTo(2f));
            Assert.That(transposedTensor[1, 1], Is.EqualTo(5f));
            Assert.That(transposedTensor[2, 0], Is.EqualTo(3f));
            Assert.That(transposedTensor[2, 1], Is.EqualTo(6f));
        }

        [Test]
        public void Transpose_WithNon2DArray_ThrowsInvalidOperationException()
        {
            // Arrange
            int[] shape = { 2, 3, 4 };
            float[] data = new float[24];
            var tensor = new Tensor(shape, data);

            // Act & Assert
            var ex = Assert.Throws<InvalidOperationException>(() =>
            {
                tensor.Transpose();
            });
            Assert.That(ex.Message, Is.EqualTo("Транспонирование поддерживается только для двумерных тензоров."));
        }
        [Test]
        public void Clip_WithValidRange_ClipsValues()
        {
            // Arrange
            int[] shape = { 2, 3 };
            float[] data = { -1f, 2f, 3f, 4f, 5f, 6f };
            var tensor = new Tensor(shape, data);

            // Act
            var clippedTensor = tensor.Clip(0f, 5f);

            // Assert
            Assert.That(clippedTensor[0, 0], Is.EqualTo(0f));  // Значение -1 должно быть ограничено 0
            Assert.That(clippedTensor[1, 2], Is.EqualTo(5f));  // Значение 6 должно быть ограничено 5
        }
        [Test]
        public void Reshape_WithValidShape_ReturnsReshapedTensor()
        {
            // Arrange
            int[] shape = { 2, 3 };
            float[] data = { 1f, 2f, 3f, 4f, 5f, 6f };
            var tensor = new Tensor(shape, data);
            int[] newShape = { 3, 2 };

            // Act
            var reshapedTensor = tensor.Reshape(newShape);

            // Assert
            Assert.That(reshapedTensor.Shape[0], Is.EqualTo(3));
            Assert.That(reshapedTensor.Shape[1], Is.EqualTo(2));
            Assert.That(reshapedTensor[0, 0], Is.EqualTo(1f));
            Assert.That(reshapedTensor[2, 1], Is.EqualTo(6f));
        }

        [Test]
        public void Reshape_WithIncompatibleShape_ThrowsInvalidOperationException()
        {
            // Arrange
            int[] shape = { 2, 3 };
            float[] data = { 1f, 2f, 3f, 4f, 5f, 6f };
            var tensor = new Tensor(shape, data);
            int[] newShape = { 4, 2 };  // Несоответствие количества элементов

            // Act & Assert
            var ex = Assert.Throws<InvalidOperationException>(() =>
            {
                tensor.Reshape(newShape);
            });
            Assert.That(ex.Message, Is.EqualTo("Новая форма должна содержать такое же количество элементов, как и старая."));
        }
        [Test]
        public void Split_WithValidAxis_ReturnsSplitTensors()
        {
            // Arrange
            int[] shape = { 4, 6 };
            float[] data = new float[24];
            var tensor = new Tensor(shape, data);

            // Act
            var splitTensors = tensor.Split(0, 2); // Разделим по оси 0 на 2 части

            // Assert
            Assert.That(splitTensors.Length, Is.EqualTo(2));
            Assert.That(splitTensors[0].Shape[0], Is.EqualTo(2));
            Assert.That(splitTensors[1].Shape[0], Is.EqualTo(2));
        }

        [Test]
        public void Split_WithInvalidAxis_ThrowsArgumentException()
        {
            // Arrange
            int[] shape = { 4, 6 };
            float[] data = new float[24];
            var tensor = new Tensor(shape, data);

            // Act & Assert
            var ex = Assert.Throws<ArgumentException>(() =>
            {
                tensor.Split(2, 2);  // Несуществующая ось
            });
            Assert.That(ex.Message, Is.EqualTo("Ось должна быть в пределах размерности тензора."));
        }

        [Test]
        public void Split_WithInvalidParts_ThrowsArgumentException()
        {
            // Arrange
            int[] shape = { 4, 6 };
            float[] data = new float[24];
            var tensor = new Tensor(shape, data);

            // Act & Assert
            var ex = Assert.Throws<ArgumentException>(() =>
            {
                tensor.Split(0, 3);  // Размерность 4 не делится на 3
            });
            Assert.That(ex.Message, Is.EqualTo("Размерность вдоль оси не делится на количество частей."));
        }
        [Test]
        public void TransposeAxes_WithValidNewOrder_ReturnsTransposedTensor()
        {
            // Arrange
            int[] shape = { 2, 3, 4 };
            float[] data = new float[24];
            var tensor = new Tensor(shape, data);
            int[] newOrder = { 2, 0, 1 };

            // Act
            var transposedTensor = tensor.TransposeAxes(newOrder);

            // Assert
            Assert.That(transposedTensor.Shape[0], Is.EqualTo(4));
            Assert.That(transposedTensor.Shape[1], Is.EqualTo(2));
            Assert.That(transposedTensor.Shape[2], Is.EqualTo(3));
        }

        [Test]
        public void TransposeAxes_WithInvalidNewOrder_ThrowsInvalidOperationException()
        {
            // Arrange
            int[] shape = { 2, 3, 4 };
            float[] data = new float[24];
            var tensor = new Tensor(shape, data);
            int[] newOrder = { 2, 2, 1 };  // Некорректный порядок

            // Act & Assert
            var ex = Assert.Throws<InvalidOperationException>(() =>
            {
                tensor.TransposeAxes(newOrder);
            });
            Assert.That(ex.Message, Is.EqualTo("Неверный порядок осей (переопределение или неправильный индекс)."));
        }

    }
}