using MetaNexus.Lib.NeuralNetwork.Tensor.Abstractions;

namespace MetaNexus.Lib.NeuralNetwork.Tensor
{
    public partial struct Tensor : ITensorMatrixOperations
    {
        // Матричное умножение двух тензоров
        public Tensor MatrixMultiply(Tensor other)
        {
            if (_shape.Length != 2 || other._shape.Length != 2)
                throw new InvalidOperationException("Для матричного умножения тензоры должны быть двухмерными.");

            if (_shape[1] != other._shape[0])
                throw new InvalidOperationException("Число столбцов первого тензора должно совпадать с числом строк второго тензора.");

            int rows = _shape[0];
            int cols = other._shape[1];
            int commonDim = _shape[1];

            var result = new Tensor(new int[] { rows, cols });

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    float sum = 0;
                    for (int k = 0; k < commonDim; k++)
                    {
                        sum += _data[i * commonDim + k] * other._data[k * cols + j];
                    }
                    result._data[i * cols + j] = sum;
                }
            }

            return result;
        }

        // Матричное деление (аналогично умножению на обратную матрицу)
        public Tensor MatrixDivide(Tensor other)
        {
            // Для деления матриц необходимо, чтобы другая матрица была квадратной и обратимой
            if (other._shape[0] != other._shape[1])
                throw new InvalidOperationException("Для матричного деления вторая матрица должна быть квадратной.");

            Tensor inverseOther = other.Inverse();
            return this.MatrixMultiply(inverseOther);
        }

        // Вычисление детерминанта матрицы (для квадратной матрицы)
        public float Determinant()
        {
            if (_shape[0] != _shape[1])
                throw new InvalidOperationException("Детерминант можно вычислить только для квадратных матриц.");

            return CalculateDeterminant(_data, _shape[0]);
        }

        // Обратная матрица (для квадратных матриц)
        public Tensor Inverse()
        {
            if (_shape[0] != _shape[1])
                throw new InvalidOperationException("Обратную матрицу можно вычислить только для квадратных матриц.");

            return CalculateInverse(_data, _shape[0]);
        }

        // Вспомогательный метод для вычисления детерминанта (через LU-разложение для квадратных матриц)
        private float CalculateDeterminant(float[] matrixData, int n)
        {
            // Используем LU-разложение для более эффективного вычисления детерминанта
            var (L, U) = LUDecompose(matrixData, n);

            float det = 1;
            for (int i = 0; i < n; i++)
            {
                det *= U[i * n + i]; // Детерминант = произведение диагональных элементов матрицы U
            }

            return det;
        }

        // Вспомогательный метод для LU-разложения матрицы
        private (float[] L, float[] U) LUDecompose(float[] matrixData, int n)
        {
            float[] L = new float[n * n];
            float[] U = new float[n * n];
            Array.Copy(matrixData, U, matrixData.Length);

            for (int i = 0; i < n; i++)
            {
                L[i * n + i] = 1; // Единичная матрица для L на диагонали
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

        // Вспомогательный метод для вычисления скалярного произведения L и U для LU-разложения
        private float DotProduct(float[] L, float[] U, int row, int col, int n)
        {
            float result = 0;
            for (int i = 0; i < row; i++)
            {
                result += L[row * n + i] * U[i * n + col];
            }
            return result;
        }

        // Вспомогательный метод для вычисления обратной матрицы через метод Гаусса с меньшими накладными расходами
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
