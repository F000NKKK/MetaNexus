using MetaNexus.Lib.NeuralNetwork.Tensors;

namespace MetaNexus.Lib.NeuralNetwork.Tests
{
    public class TensorMatrixOperationsTests
    {
        [SetUp]
        public void Setup()
        {
        }

        [Test]
        public void Test_Inverse_ValidSquareMatrix_ReturnsCorrectInverse()
        {
            // Исходная матрица 2x2
            var tensor = new Tensor(new int[] { 2, 2 }, new float[] { 4, 7, 2, 6 });

            // Ожидаемая обратная матрица
            var expectedInverse = new Tensor(new int[] { 2, 2 }, new float[] { 0.6f, -0.7f, -0.2f, 0.4f });

            // Вычисление обратной матрицы
            var result = tensor.Inverse();

            // Проверка, что результат соответствует ожидаемой обратной матрице
            Assert.That(result.Data, Is.EqualTo(expectedInverse.Data).Within(0.0001f));  // Допускаем погрешность
        }

        [Test]
        public void Test_Inverse_NonSquareMatrix_ThrowsException()
        {
            // Неквадратная матрица 2x3
            var tensor = new Tensor(new int[] { 2, 3 }, new float[] { 1, 2, 3, 4, 5, 6 });

            // Проверка, что инверсия выбрасывает исключение для неквадратной матрицы
            Assert.Throws<InvalidOperationException>(() => tensor.Inverse());
        }

        [Test]
        public void Test_Inverse_MultiplyByInverse_GivesIdentityMatrix()
        {
            // Исходная матрица 2x2
            var tensor = new Tensor(new int[] { 2, 2 }, new float[] { 4, 7, 2, 6 });

            // Вычисление обратной матрицы
            var inverseTensor = tensor.Inverse();

            // Умножение исходной матрицы на её обратную
            var result = tensor.Dot(inverseTensor);

            // Ожидаемая единичная матрица
            var identityMatrix = new Tensor(new int[] { 2, 2 }, new float[] { 1f, 0f, 0f, 1f });

            // Проверка, что результат умножения равен единичной матрице
            Assert.That(result.Data, Is.EqualTo(identityMatrix.Data).Within(0.0001f));  // Допускаем погрешность
        }

        [Test]
        public void Test_Dot_MatrixMultiplication()
        {
            // Умножение 2x3 матрицы на 3x2 матрицу
            var tensor1 = new Tensor(new int[] { 2, 3 }, new float[] { 1, 2, 3, 4, 5, 6 });
            var tensor2 = new Tensor(new int[] { 3, 2 }, new float[] { 7, 8, 9, 10, 11, 12 });

            var result = tensor1.Dot(tensor2);

            // Ожидаемый результат: 2x2 матрица
            var expectedData = new float[] { 58, 64, 139, 154 }; // Результат умножения
            Assert.That(result.Data, Is.EqualTo(expectedData));
        }

        [Test]
        public void Test_MatrixDivide_DivideByMatrix()
        {
            // Деление 2x2 матрицы на обратную матрицу 2x2
            var tensor1 = new Tensor(new int[] { 2, 2 }, new float[] { 1, 2, 3, 4 });
            var tensor2 = new Tensor(new int[] { 2, 2 }, new float[] { 5, 6, 7, 8 });

            var result = tensor1.MatrixDivide(tensor2);

            // Ожидаемый результат после деления (умножения на обратную матрицу tensor2)
            var expectedData = new float[] { 3.0f, -2.0f, 2.0f, -1.0f };
            Assert.That(result.Data, Is.EqualTo(expectedData).Within(0.0001f)); // Допускаем ошибку погрешности
        }

        [Test]
        public void Test_Determinant_2x2Matrix()
        {
            // Детерминант для 2x2 матрицы
            var tensor = new Tensor(new int[] { 2, 2 }, new float[] { 1, 2, 3, 4 });

            var result = tensor.Determinant();

            var expectedDeterminant = -2.0f; // Детерминант 2x2 матрицы
            Assert.That(result, Is.EqualTo(expectedDeterminant));
        }

        [Test]
        public void Test_Inverse_2x2Matrix()
        {
            // Обратная матрица для 2x2 матрицы
            var tensor = new Tensor(new int[] { 2, 2 }, new float[] { 4, 7, 2, 6 });

            var result = tensor.Inverse();

            // Ожидаемая обратная матрица
            var expectedData = new float[] { 0.6f, -0.7f, -0.2f, 0.4f };
            Assert.That(result.Data, Is.EqualTo(expectedData).Within(0.0001f)); // Допускаем погрешность
        }

        [Test]
        public void Test_Dot_MatrixMultiplication_InvalidDimensions()
        {
            // Проверка умножения матриц с несовпадающими размерами
            var tensor1 = new Tensor(new int[] { 2, 3 }, new float[] { 1, 2, 3, 4, 5, 6 });
            var tensor2 = new Tensor(new int[] { 2, 2 }, new float[] { 7, 8, 9, 10 });

            Assert.Throws<InvalidOperationException>(() => tensor1.Dot(tensor2), "Число столбцов первого тензора должно совпадать с числом строк второго тензора.");
        }

        [Test]
        public void Test_Determinant_InvalidMatrix()
        {
            // Проверка детерминанта для не квадратной матрицы
            var tensor = new Tensor(new int[] { 2, 3 }, new float[] { 1, 2, 3, 4, 5, 6 });

            Assert.Throws<InvalidOperationException>(() => tensor.Determinant(), "Детерминант можно вычислить только для квадратных матриц.");
        }

        [Test]
        public void Test_Inverse_InvalidMatrix()
        {
            // Проверка обратной матрицы для не квадратной матрицы
            var tensor = new Tensor(new int[] { 2, 3 }, new float[] { 1, 2, 3, 4, 5, 6 });

            Assert.Throws<InvalidOperationException>(() => tensor.Inverse(), "Обратную матрицу можно вычислить только для квадратных матриц.");
        }
    }
}
