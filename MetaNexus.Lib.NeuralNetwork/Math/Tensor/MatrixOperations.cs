using System.Numerics;

namespace MetaNexus.Lib.NeuralNetwork.Math.Tensor
{
    /// <summary>
    /// Класс, предоставляющий основные операции для работы с матрицами.
    /// </summary>
    public static class MatrixOperations
    {
        /// <summary>
        /// Выполняет умножение двух матриц.
        /// </summary>
        /// <param name="a">Первая матрица (тензор).</param>
        /// <param name="b">Вторая матрица (тензор).</param>
        /// <returns>Результат матричного умножения.</returns>
        public static Tensor<T> Multiply<T>(Tensor<T> a, Tensor<T> b) where T : INumber<T>
        {
            // Убедимся, что количество столбцов первой матрицы равно количеству строк второй матрицы
            if (a.Shape[1] != b.Shape[0])
                throw new ArgumentException("Number of columns of the first matrix must be equal to number of rows of the second matrix.");

            int rows = a.Shape[0];
            int cols = b.Shape[1];

            // Матричное умножение через LINQ
            T?[] resultData = Enumerable.Range(0, rows)
                .Select(i => Enumerable.Range(0, cols)
                    .Select(j => Enumerable.Range(0, a.Shape[1])
                        .Aggregate(default(T), (acc, k) =>
                        {
                            return acc + a.GetValue(i, k) * b.GetValue(k, j);
                        })
                    ).ToArray()
                ).SelectMany(x => x).ToArray(); // Преобразование вложенных массивов в одномерный результат

            return new Tensor<T>(resultData as T[], rows, cols);
        }

        /// <summary>
        /// Выполняет транспонирование матрицы.
        /// </summary>
        /// <param name="a">Тензор для транспонирования.</param>
        /// <returns>Тензор, являющийся транспонированным вариантом исходного.</returns>
        public static Tensor<T> Transpose<T>(Tensor<T> a) where T : INumber<T>
        {
            if (a.Rank != 2)
                throw new InvalidOperationException("Transpose is only supported for 2D tensors.");

            int rows = a.Shape[0];
            int cols = a.Shape[1];

            // Транспонирование через LINQ
            T[] resultData = Enumerable.Range(0, cols)
                .SelectMany(j => Enumerable.Range(0, rows)
                    .Select(i => a.GetValue(i, j))
                ).ToArray();

            return new Tensor<T>(resultData, cols, rows);
        }

        /// <summary>
        /// Вычисляет детерминант матрицы.
        /// </summary>
        /// <param name="a">Квадратная матрица.</param>
        /// <returns>Детерминант матрицы.</returns>
        public static T Determinant<T>(Tensor<T> a) where T : INumber<T>
        {
            if (a.Shape[0] != a.Shape[1])
                throw new ArgumentException("Matrix must be square to compute determinant.");

            int n = a.Shape[0];

            // Для 2x2 матрицы: det(A) = a11 * a22 - a12 * a21
            if (n == 2)
            {
                var det = Enumerable.Range(0, 2)
                    .Select(i => a.GetValue(i, 0) * a.GetValue(i, 1))
                    .Aggregate((x, y) => x - y);

                return det;
            }

            throw new NotImplementedException("Determinant calculation for matrices larger than 2x2 is not implemented.");
        }

        /// <summary>
        /// Находит обратную матрицу для квадратной матрицы.
        /// </summary>
        /// <param name="a">Квадратная матрица.</param>
        /// <returns>Обратная матрица.</returns>
        public static Tensor<T> Inverse<T>(Tensor<T> a) where T : INumber<T>
        {
            if (a.Shape[0] != a.Shape[1])
                throw new ArgumentException("Matrix must be square to compute inverse.");

            int n = a.Shape[0];

            // Для 2x2 матрицы
            if (n == 2)
            {
                T a11 = a.GetValue(0, 0);
                T a12 = a.GetValue(0, 1);
                T a21 = a.GetValue(1, 0);
                T a22 = a.GetValue(1, 1);
                T det = a11 * a22 - a12 * a21;


                if (det.Equals(default(T))) // Если детерминант равен 0, матрица необратима
                    throw new InvalidOperationException("Matrix is not invertible.");

                T invDet = (T.One / det); // Обратный детерминант

                // Формула для обратной матрицы 2x2
                var resultData = new T[]
                {
                    a22 * invDet, -a12 * invDet,
                    -a21 * invDet, a11 * invDet
                };

                return new Tensor<T>(resultData, 2, 2);
            }

            throw new NotImplementedException("Inverse calculation for matrices larger than 2x2 is not implemented.");
        }
    }
}