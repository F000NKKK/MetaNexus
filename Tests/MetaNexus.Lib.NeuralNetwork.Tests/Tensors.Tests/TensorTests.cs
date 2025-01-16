using MetaNexus.Lib.NeuralNetwork.Tensors;

namespace MetaNexus.Lib.NeuralNetwork.Tests
{
    public class TensorTests
    {
        [SetUp]
        public void Setup()
        {
        }

        [Test]
        public void Tensor_Constructor_WithShape_CreatesTensorWithCorrectSize()
        {
            // Arrange
            int[] shape = { 3, 4 };

            // Act
            Tensor tensor = new Tensor(shape);

            // Assert
            // Проверяем, что размер тензора соответствует ожидаемому (3 * 4 = 12)
            Assert.That(tensor.Size, Is.EqualTo(12));
            // Проверяем, что тензор имеет два измерения
            Assert.That(tensor.Shape.Length, Is.EqualTo(2));
            // Проверяем, что размерность по первому измерению равна 3
            Assert.That(tensor.Shape[0], Is.EqualTo(3));
            // Проверяем, что размерность по второму измерению равна 4
            Assert.That(tensor.Shape[1], Is.EqualTo(4));
        }

        [Test]
        public void Tensor_Constructor_WithShapeAndData_CreatesTensorWithCorrectData()
        {
            // Arrange
            int[] shape = { 2, 2 };
            float[] data = { 1.0f, 2.0f, 3.0f, 4.0f };

            // Act
            Tensor tensor = new Tensor(shape, data);

            // Assert
            // Проверяем, что данные в тензоре соответствуют заданным значениям
            Assert.That(tensor[0, 0], Is.EqualTo(1.0f));
            Assert.That(tensor[0, 1], Is.EqualTo(2.0f));
            Assert.That(tensor[1, 0], Is.EqualTo(3.0f));
            Assert.That(tensor[1, 1], Is.EqualTo(4.0f));
        }

        [Test]
        public void Tensor_Indexing_CorrectlyAccessesElements()
        {
            // Arrange
            int[] shape = { 2, 3 };
            float[] data = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
            Tensor tensor = new Tensor(shape, data);

            // Act & Assert
            // Проверяем доступ к элементам через индексацию
            Assert.That(tensor[0, 0], Is.EqualTo(1.0f));
            Assert.That(tensor[0, 1], Is.EqualTo(2.0f));
            Assert.That(tensor[1, 2], Is.EqualTo(6.0f));
            Assert.That(tensor[1, 0], Is.EqualTo(4.0f));
        }

        [Test]
        public void Tensor_ApplyFunction_TransformsDataCorrectly()
        {
            // Arrange
            int[] shape = { 2, 2 };
            float[] data = { 1.0f, 2.0f, 3.0f, 4.0f };
            Tensor tensor = new Tensor(shape, data);

            // Act
            Tensor transformedTensor = tensor.Apply(x => x * 2);

            // Assert
            // Проверяем, что данные в тензоре были правильно преобразованы (умножены на 2)
            Assert.That(transformedTensor[0, 0], Is.EqualTo(2.0f));
            Assert.That(transformedTensor[0, 1], Is.EqualTo(4.0f));
            Assert.That(transformedTensor[1, 0], Is.EqualTo(6.0f));
            Assert.That(transformedTensor[1, 1], Is.EqualTo(8.0f));
        }

        [Test]
        public void Tensor_Clone_CreatesIndependentCopy()
        {
            // Arrange
            int[] shape = { 2, 2 };
            float[] data = { 1.0f, 2.0f, 3.0f, 4.0f };
            Tensor tensor = new Tensor(shape, data);

            // Act
            Tensor clonedTensor = tensor.Clone();
            clonedTensor[0, 0] = 100.0f;

            // Assert
            // Проверяем, что оригинальный тензор и клонированный тензор независимы
            Assert.That(tensor[0, 0], Is.EqualTo(1.0f));
            Assert.That(clonedTensor[0, 0], Is.EqualTo(100.0f));
            Assert.That(tensor[0, 0], Is.Not.EqualTo(clonedTensor[0, 0]));
        }

        [Test]
        public void Tensor_Flatten_Creates1DArray()
        {
            // Arrange
            int[] shape = { 2, 3 };
            float[] data = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
            Tensor tensor = new Tensor(shape, data);

            // Act
            Tensor flattenedTensor = tensor.Flatten();

            // Assert
            // Проверяем, что после преобразования в одномерный тензор его форма содержит одно измерение
            Assert.That(flattenedTensor.Shape.Length, Is.EqualTo(1));
            // Проверяем, что размерность одномерного тензора соответствует общей численности элементов
            Assert.That(flattenedTensor.Shape[0], Is.EqualTo(6)); // 2 * 3 = 6
            Assert.That(flattenedTensor[0], Is.EqualTo(1.0f));
            Assert.That(flattenedTensor[5], Is.EqualTo(6.0f));
        }

        [Test]
        public void Tensor_Constructor_ThrowsExceptionForTooLargeTensorSize()
        {
            // Arrange
            int[] shape = { int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue };

            // Act & Assert
            // Проверяем, что конструктор выбрасывает исключение для слишком большого размера тензора
            Assert.That(() => new Tensor(shape), Throws.TypeOf<InvalidOperationException>());
        }

        [Test]
        public void Tensor_Indexing_ThrowsExceptionForInvalidIndices()
        {
            // Arrange
            int[] shape = { 2, 2 };
            float[] data = { 1.0f, 2.0f, 3.0f, 4.0f };
            Tensor tensor = new Tensor(shape, data);

            // Act & Assert
            // Проверяем, что выбрасывается исключение при попытке индексации за пределами массива
            Assert.That(() => tensor[3, 0], Throws.TypeOf<ArgumentException>());
            Assert.That(() => tensor[0, 2], Throws.TypeOf<ArgumentException>());
        }

        // Тест для исключения, если размер данных не соответствует размеру тензора
        [Test]
        public void Constructor_WithShapeAndData_ThrowsExceptionWhenDataSizeIsIncorrect()
        {
            // Arrange
            int[] shape = { 2, 3 };
            float[] data = { 1f, 2f }; // Недостаточное количество данных

            // Act & Assert
            var ex = Assert.Throws<ArgumentException>(() => new Tensor(shape, data));
            Assert.That(ex.Message, Is.EqualTo("Размер массива данных не соответствует размеру тензора."));
        }

        // Тест для исключения, если shape равен null
        [Test]
        public void Constructor_WithNullShape_ThrowsArgumentNullException()
        {
            // Arrange
            float[] data = { 1f, 2f, 3f };

            // Act & Assert
#pragma warning disable CS8625 // Литерал, равный NULL, не может быть преобразован в ссылочный тип, не допускающий значение NULL.
            var ex = Assert.Throws<ArgumentNullException>(() => new Tensor(null, data));
#pragma warning restore CS8625 // Литерал, равный NULL, не может быть преобразован в ссылочный тип, не допускающий значение NULL.
            Assert.That(ex.ParamName, Is.EqualTo("shape"));
        }

        // Тест для исключения, если data равен null
        [Test]
        public void Constructor_WithNullData_ThrowsArgumentNullException()
        {
            // Arrange
            int[] shape = { 2, 3 };

            // Act & Assert
#pragma warning disable CS8625 // Литерал, равный NULL, не может быть преобразован в ссылочный тип, не допускающий значение NULL.
            var ex = Assert.Throws<ArgumentNullException>(() => new Tensor(shape, null));
#pragma warning restore CS8625 // Литерал, равный NULL, не может быть преобразован в ссылочный тип, не допускающий значение NULL.
            Assert.That(ex.ParamName, Is.EqualTo("data"));
        }

        // Тест для конструктора копирования (глубокое копирование)
        [Test]
        public void Constructor_WithExistingTensor_CreatesDeepCopy()
        {
            // Arrange
            int[] shape = { 2, 2 };
            float[] data = { 1f, 2f, 3f, 4f };
            var originalTensor = new Tensor(shape, data);

            // Act
            var copiedTensor = new Tensor(originalTensor);

            // Assert
            Assert.That(originalTensor.Shape, Is.EqualTo(copiedTensor.Shape));
            Assert.That(originalTensor.Size, Is.EqualTo(copiedTensor.Size));
            Assert.That(originalTensor.Rank, Is.EqualTo(copiedTensor.Rank));
            Assert.That(originalTensor.Data.Span.ToArray(), Is.EqualTo(copiedTensor.Data.Span.ToArray()));
            Assert.That(copiedTensor.Data.Span.ToArray(), Is.EqualTo(data));
            for (int i = 0; i < shape[0]; i++)
            {
                for (int j = 0; j < shape[1]; j++)
                {
                    Assert.That(originalTensor[i, j], Is.EqualTo(copiedTensor[i, j]));
                }
            }

            // Изменим данные в оригинальном тензоре и проверим, что копия не изменилась
            originalTensor[0, 0] = 999f;
            Assert.That(originalTensor[0, 0], Is.Not.EqualTo(copiedTensor[0, 0]));
        }

        [Test]
        public void Indexer_WithIncorrectNumberOfIndices_ThrowsArgumentException()
        {
            // Arrange
            int[] shape = { 2, 3 };
            float[] data = { 1f, 2f, 3f, 4f, 5f, 6f };
            var tensor = new Tensor(shape, data);

            // Act & Assert
            var ex = Assert.Throws<ArgumentException>(() =>
            {
                // Попытка использовать один индекс при тензоре с двумя измерениями
                float a = tensor[0]; // Один индекс вместо двух
            });

            Assert.That(ex.Message, Is.EqualTo("Количество индексов не соответствует рангу тензора."));
        }

        [Test]
        public void Indexer_WithOutOfRangeIndices_ThrowsArgumentException()
        {
            // Arrange
            int[] shape = { 2, 3 };
            float[] data = { 1f, 2f, 3f, 4f, 5f, 6f };
            var tensor = new Tensor(shape, data);

            // Act & Assert
            var ex = Assert.Throws<ArgumentException>(() =>
            {
                // Попытка использовать индекс, выходящий за пределы размерности
                float a = tensor[2, 1]; // Индекс 2 для первого измерения выходит за пределы (только 0 и 1 допустимы)
            });

            Assert.That(ex.Message, Is.EqualTo("Индексы выходят за пределы массива."));
        }


        // Тест для многомерной индексации через params int[] indices
        [Test]
        public void Indexer_WithMultiDimensionalIndices_ReturnsCorrectValue()
        {
            // Arrange
            int[] shape = { 2, 3 };
            float[] data = { 1f, 2f, 3f, 4f, 5f, 6f };
            var tensor = new Tensor(shape, data);

            // Act & Assert
            Assert.That(tensor[0, 0], Is.EqualTo(1f));
            Assert.That(tensor[1, 0], Is.EqualTo(4f));
            Assert.That(tensor[0, 1], Is.EqualTo(2f));
            Assert.That(tensor[1, 2], Is.EqualTo(6f));
        }

        // Тест для выброса исключения при индексах, выходящих за пределы массива
        [Test]
        public void Indexer_WithOutOfBoundsIndices_ThrowsArgumentException()
        {
            // Arrange
            int[] shape = { 2, 3 };
            float[] data = { 1f, 2f, 3f, 4f, 5f, 6f };
            var tensor = new Tensor(shape, data);

            // Act & Assert
            var ex = Assert.Throws<ArgumentException>(() =>
            {
                float a = tensor[2, 0];
            }); // Индекс вне допустимого диапазона
            Assert.That(ex.Message, Is.EqualTo("Индексы выходят за пределы массива."));
        }

        // Тест для FlattenFloatArray, чтобы проверить правильность возвращаемого массива
        [Test]
        public void FlattenFloatArray_ReturnsCorrectArray()
        {
            // Arrange
            int[] shape = { 2, 3 };
            float[] data = { 1f, 2f, 3f, 4f, 5f, 6f };
            var tensor = new Tensor(shape, data);

            // Act
            float[] flattened = tensor.FlattenFloatArray();

            // Assert
            Assert.That(flattened.Length, Is.EqualTo(data.Length));
            for (int i = 0; i < data.Length; i++)
            {
                Assert.That(flattened[i], Is.EqualTo(data[i]));
            }
        }

        // Тест для IsEmpty, чтобы проверить, что метод правильно определяет пустой тензор
        [Test]
        public void IsEmpty_ReturnsTrueForEmptyTensor()
        {
            // Arrange
            int[] shape = { 0, 0 };
            var tensor = new Tensor(shape);

            // Act & Assert
            Assert.That(tensor.IsEmpty(), Is.True);
        }

        // Тест для IsEmpty, чтобы проверить, что метод правильно определяет непустой тензор
        [Test]
        public void IsEmpty_ReturnsFalseForNonEmptyTensor()
        {
            // Arrange
            int[] shape = { 2, 3 };
            float[] data = { 1f, 2f, 3f, 4f, 5f, 6f };
            var tensor = new Tensor(shape, data);

            // Act & Assert
            Assert.That(tensor.IsEmpty(), Is.False);
        }
    }
}