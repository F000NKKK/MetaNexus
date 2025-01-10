using System.Numerics;

namespace MetaNexus.Lib.NeuralNetwork.Math.Tensor
{
    /// <summary>
    /// Класс, предоставляющий основные математические операции для работы с тензорами.
    /// </summary>
    public static class TensorMath
    {
        /// <summary>
        /// Масштабирует все элементы тензора на заданное скалярное значение.
        /// </summary>
        /// <param name="scalar">Скаляр для умножения.</param>
        /// <param name="tensor">Тензор для масштабирования.</param>
        /// <returns>Новый масштабированный тензор.</returns>
        public static Tensor<T> Scale<T>(this Tensor<T> tensor, T scalar) where T : INumber<T>
        {
            var resultData = tensor.Data.Select(x => x * scalar).ToArray();
            return new Tensor<T>(resultData, tensor.Shape);
        }

        /// <summary>
        /// Вычисляет сумму всех элементов тензора.
        /// </summary>
        /// <param name="tensor">Тензор, элементы которого суммируются.</param>
        /// <returns>Сумма элементов.</returns>
        public static T Sum<T>(this Tensor<T> tensor) where T : INumber<T>
        {
            return tensor.Data.Aggregate(T.Zero, (acc, x) => acc + x);
        }

        /// <summary>
        /// Находит максимальное значение среди всех элементов тензора.
        /// </summary>
        /// <param name="tensor">Тензор, из которого извлекается максимальное значение.</param>
        /// <returns>Максимальное значение.</returns>
        public static T? Max<T>(this Tensor<T> tensor) where T : INumber<T>
        {
            return tensor.Data.Max();
        }

        /// <summary>
        /// Находит минимальное значение среди всех элементов тензора.
        /// </summary>
        /// <param name="tensor">Тензор, из которого извлекается минимальное значение.</param>
        /// <returns>Минимальное значение.</returns>
        public static T? Min<T>(this Tensor<T> tensor) where T : INumber<T>
        {
            return tensor.Data.Min();
        }

        /// <summary>
        /// Преобразует тензор в новую форму.
        /// </summary>
        /// <param name="tensor">Тензор, который требуется изменить.</param>
        /// <param name="newShape">Новая форма.</param>
        /// <returns>Тензор с новой формой.</returns>
        public static Tensor<T> Reshape<T>(this Tensor<T> tensor, params int[] newShape) where T : INumber<T>
        {
            if (newShape.Aggregate(1, (a, b) => a * b) != tensor.Data.Length)
                throw new ArgumentException("New shape must have the same total number of elements as the original.");
            return new Tensor<T>(tensor.Data, newShape);
        }

        /// <summary>
        /// Выполняет поэлементное сложение двух тензоров.
        /// </summary>
        /// <param name="a">Первый тензор.</param>
        /// <param name="b">Второй тензор.</param>
        /// <returns>Новый тензор, являющийся результатом сложения.</returns>
        /// <exception cref="ArgumentException">Вызывается, если размеры тензоров не совпадают.</exception>
        public static Tensor<T> Add<T>(Tensor<T> a, Tensor<T> b) where T : INumber<T>
        {
            if (!a.Shape.SequenceEqual(b.Shape))
                throw new ArgumentException("Shape of tensors must match for addition.");

            var resultData = a.Data.Zip(b.Data, (x, y) => x + y).ToArray();
            return new Tensor<T>(resultData, a.Shape);
        }

        /// <summary>
        /// Выполняет поэлементное вычитание одного тензора из другого.
        /// </summary>
        /// <param name="a">Первый тензор.</param>
        /// <param name="b">Второй тензор.</param>
        /// <returns>Новый тензор, являющийся результатом вычитания.</returns>
        /// <exception cref="ArgumentException">Вызывается, если размеры тензоров не совпадают.</exception>


        public static Tensor<T> Subtract<T>(Tensor<T> a, Tensor<T> b) where T : INumber<T>
        {
            if (!a.Shape.SequenceEqual(b.Shape))
                throw new ArgumentException("Shape of tensors must match for subtraction.");

            var resultData = a.Data.Zip(b.Data, (x, y) => x - y).ToArray();
            return new Tensor<T>(resultData, a.Shape);
        }

        /// <summary>
        /// Выполняет поэлементное умножение двух тензоров.
        /// </summary>
        /// <param name="a">Первый тензор.</param>
        /// <param name="b">Второй тензор.</param>
        /// <returns>Новый тензор, являющийся результатом умножения.</returns>
        /// <exception cref="ArgumentException">Вызывается, если размеры тензоров не совпадают.</exception>
        public static Tensor<T> Multiply<T>(Tensor<T> a, Tensor<T> b) where T : INumber<T>
        {
            if (!a.Shape.SequenceEqual(b.Shape))
                throw new ArgumentException("Shape of tensors must match for multiplication.");

            var resultData = a.Data.Zip(b.Data, (x, y) => x * y).ToArray();
            return new Tensor<T>(resultData, a.Shape);
        }

        /// <summary>
        /// Выполняет поэлементное деление одного тензора на другой.
        /// </summary>
        /// <param name="a">Первый тензор.</param>
        /// <param name="b">Второй тензор.</param>
        /// <returns>Новый тензор, являющийся результатом деления.</returns>
        /// <exception cref="ArgumentException">Вызывается, если размеры тензоров не совпадают.</exception>
        public static Tensor<T> Divide<T>(Tensor<T> a, Tensor<T> b) where T : INumber<T>
        {
            if (!a.Shape.SequenceEqual(b.Shape))
                throw new ArgumentException("Shape of tensors must match for division.");

            var resultData = a.Data.Zip(b.Data, (x, y) =>
            {
                if (y.Equals(default(T)))
                    throw new DivideByZeroException("Division by zero encountered.");
                return x / y;
            }).ToArray();
            return new Tensor<T>(resultData, a.Shape);
        }

        /// <summary>
        /// Выполняет поэлементное возведение в степень каждого элемента тензора.
        /// </summary>
        /// <param name="a">Тензор.</param>
        /// <param name="exponent">Степень, в которую возводятся элементы тензора.</param>
        /// <returns>Новый тензор, являющийся результатом возведения в степень.</returns>
        public static Tensor<T> Power<T>(Tensor<T> a, T exponent) where T : INumber<T>
        {
            // Преобразуем степень в double для работы с дробными значениями
            var exponentDouble = Convert.ToDouble(exponent);

            var resultData = a.Data.Select(x =>
            {
                double value = Convert.ToDouble(x); // Преобразуем каждый элемент в double

                double result = double.Pow(value, exponentDouble);

                return (T)Convert.ChangeType(result, typeof(T));
            }).ToArray();

            return new Tensor<T>(resultData, a.Shape);
        }

        /// <summary>
        /// Вычисляет скалярное произведение двух тензоров.
        /// </summary>
        /// <param name="a">Первый тензор.</param>
        /// <param name="b">Второй тензор.</param>
        /// <returns>Скалярное произведение.</returns>
        /// <exception cref="ArgumentException">Вызывается, если размеры тензоров не совпадают.</exception>
        public static T Dot<T>(Tensor<T> a, Tensor<T> b) where T : INumber<T>
        {
            if (!a.Shape.SequenceEqual(b.Shape))
                throw new ArgumentException("Shape of tensors must match for dot product.");

            return a.Data.Zip(b.Data, (x, y) => x * y).Aggregate(T.Zero, (acc, x) => acc + x);
        }
    }
}