﻿using MetaNexus.Lib.NeuralNetwork.Tensors;

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
            int[] shape = { int.MaxValue, 2 }; // Размер тензора слишком велик

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
    }
}