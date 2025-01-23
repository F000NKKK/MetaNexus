using MetaNexus.Lib.NeuralNetwork.Tensors.Abstractions;

namespace MetaNexus.Lib.NeuralNetwork.Tensors
{
    public partial struct Tensor : ITensorMatrixOperations
    {
        public Tensor Dot(Tensor other)
        {
            if (_shape.Length != 2 || other._shape.Length != 2)
                throw new InvalidOperationException("Для матричного умножения тензоры должны быть двухмерными.");

            if (_shape[1] != other._shape[0])
                throw new InvalidOperationException("Число столбцов первого тензора должно совпадать с числом строк второго тензора.");

            int rows = _shape[0];
            int cols = other._shape[1];
            int commonDim = _shape[1];

            var resultData = new float[rows * cols];
            var resultMemory = new Memory<float>(resultData);

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    float sum = 0;
                    for (int k = 0; k < commonDim; k++)
                    {
                        sum += _data.Span[i * commonDim + k] * other._data.Span[k * cols + j];
                    }
                    resultMemory.Span[i * cols + j] = sum;
                }
            }

            // Возвращаем новый тензор с результатом умножения
            return new Tensor(new int[] { rows, cols }, resultMemory);
        }

        public Tensor MatrixDivide(Tensor other)
        {
            Console.WriteLine("Начало операции матричного деления.");

            // Проверка, что вторая матрица квадратная
            if (other._shape[0] != other._shape[1])
            {
                throw new InvalidOperationException("Для матричного деления вторая матрица должна быть квадратной.");
            }
            Console.WriteLine($"Вторая матрица (other): {_shape[0]}x{_shape[1]}");

            // Выполнение инверсии второй матрицы
            Console.WriteLine("Вычисление обратной матрицы для второй матрицы.");
            Tensor inverseOther = other.Inverse();

            // Логирование данных инверсии
            Console.WriteLine($"Данные инверсии второй матрицы (inverseOther):");
            Console.WriteLine(inverseOther.ToString());

            // Умножение текущей матрицы на инвертированную матрицу
            Console.WriteLine("Выполнение умножения текущей матрицы на обратную матрицу второй.");
            Tensor result = this.Dot(inverseOther);

            // Логирование результата умножения
            Console.WriteLine("Результат матричного деления (умножения на обратную матрицу):");
            Console.WriteLine(result.ToString());

            return result;
        }

        public float Determinant()
        {
            if (_shape[0] != _shape[1])
                throw new InvalidOperationException("Детерминант можно вычислить только для квадратных матриц.");

            return CalculateDeterminant(_data.Span.ToArray(), _shape[0]);
        }

        public Tensor Inverse()
        {
            if (_shape[0] != _shape[1])
            {
                Console.WriteLine("Ошибка: Обратную матрицу можно вычислить только для квадратных матриц.");
                throw new InvalidOperationException("Обратную матрицу можно вычислить только для квадратных матриц.");
            }

            var inverseTensor = CalculateInverse(_data.Span.ToArray(), _shape[0]);
           
            return inverseTensor;
        }

        private float CalculateDeterminant(float[] matrixData, int n)
        {
            var (L, U) = LUDecompose(matrixData, n);

            float det = 1;
            for (int i = 0; i < n; i++)
            {
                det *= U[i * n + i];
            }

            return det;
        }

        private (float[] L, float[] U) LUDecompose(float[] matrixData, int n)
        {
            float[] L = new float[n * n];
            float[] U = new float[n * n];

            Array.Copy(matrixData, U, matrixData.Length);

            for (int i = 0; i < n; i++)
            {
                L[i * n + i] = 1;
            }

            for (int i = 0; i < n; i++)
            {
                for (int j = i; j < n; j++)
                {
                    U[i * n + j] -= DotProduct(L, U, i, j, n);
                }

                for (int j = i + 1; j < n; j++)
                {
                    L[j * n + i] = (matrixData[j * n + i] - DotProduct(L, U, j, i, n)) / U[i * n + i];
                }
            }

            return (L, U);
        }

        private float DotProduct(float[] L, float[] U, int row, int col, int n)
        {
            float result = 0;
            for (int i = 0; i < row; i++)
            {
                result += L[row * n + i] * U[i * n + col];
            }
            return result;
        }

        private Tensor CalculateInverse(float[] matrixData, int n)
        {
            var matrix = (float[])matrixData.Clone();
            var identity = new float[n * n];
            for (int i = 0; i < n; i++)
            {
                identity[i * n + i] = 1;
            }

            for (int i = 0; i < n; i++)
            {
                float pivot = matrix[i * n + i];
                for (int j = 0; j < n; j++)
                {
                    matrix[i * n + j] /= pivot;
                    identity[i * n + j] /= pivot;
                }

                for (int j = 0; j < n; j++)
                {
                    if (j == i) continue;
                    float factor = matrix[j * n + i];
                    for (int k = 0; k < n; k++)
                    {
                        matrix[j * n + k] -= factor * matrix[i * n + k];
                        identity[j * n + k] -= factor * identity[i * n + k];
                    }
                }
            }

            return new Tensor(new int[] { n, n }, identity);
        }
    }
}
