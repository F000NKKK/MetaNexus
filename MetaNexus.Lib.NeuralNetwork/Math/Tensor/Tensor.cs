using System.Numerics;

namespace MetaNexus.Lib.NeuralNetwork.Math.Tensor
{
    /// <summary>
    /// Класс Tensor представляет многомерный массив числовых данных с поддержкой операций над тензорами.
    /// </summary>
    public class Tensor<T> where T : INumber<T>
    {
        /// <summary>
        /// Данные тензора в виде одномерного массива.
        /// </summary>
        public T[] Data { get; private set; }

        /// <summary>
        /// Форма (размерность) тензора.
        /// </summary>
        public int[] Shape { get; private set; }

        /// <summary>
        /// Шаги (strides) для вычисления плоского индекса.
        /// </summary>
        public int[] Strides { get; private set; }

        /// <summary>
        /// Ранг (число осей) тензора.
        /// </summary>
        public int Rank => Shape.Length;

        /// <summary>
        /// Создает новый тензор с заданной формой.
        /// </summary>
        /// <param name="shape">Форма тензора (размеры по каждой оси).</param>
        public Tensor(params int[] shape)
        {
            if (shape == null || shape.Length == 0)
                throw new ArgumentException("Shape must be non-empty and non-null.");

            Shape = shape;
            Strides = ComputeStrides(shape);
            Data = new T[Shape.Aggregate(1, (a, b) => a * b)];
        }

        /// <summary>
        /// Создает новый тензор с данными и заданной формой.
        /// </summary>
        /// <param name="data">Одномерный массив данных.</param>
        /// <param name="shape">Форма тензора (размеры по каждой оси).</param>
        public Tensor(T[] data, params int[] shape)
        {
            if (data.Length != shape.Aggregate(1, (a, b) => a * b))
                throw new ArgumentException("Data size does not match shape size.");

            Data = data;
            Shape = shape;
            Strides = ComputeStrides(shape);
        }

        /// <summary>
        /// Вычисляет шаги (strides) для доступа к элементам тензора.
        /// </summary>
        /// <param name="shape">Форма тензора.</param>
        /// <returns>Массив шагов.</returns>
        private int[] ComputeStrides(int[] shape)
        {
            return shape
                .Reverse()
                .Aggregate((strides: new List<int>(), stride: 1), (acc, dim) =>
                {
                    acc.strides.Add(acc.stride);
                    return (acc.strides, acc.stride * dim);
                })
                .strides
                .Reverse<int>()
                .ToArray();
        }

        /// <summary>
        /// Преобразует многомерные индексы в плоский индекс.
        /// </summary>
        /// <param name="indices">Многомерные индексы.</param>
        /// <returns>Плоский индекс.</returns>
        private int GetFlatIndex(params int[] indices)
        {
            if (indices.Length != Rank)
                throw new ArgumentException("Number of indices must match the rank of the tensor.");

            return indices
                .Select((index, axis) =>
                {
                    if (index < 0 || index >= Shape[axis])
                        throw new IndexOutOfRangeException($"Index {index} is out of bounds for axis {axis}.");
                    return index * Strides[axis];
                })
                .Sum();
        }

        /// <summary>
        /// Получает значение элемента по многомерным индексам.
        /// </summary>
        /// <param name="indices">Многомерные индексы.</param>
        /// <returns>Значение элемента.</returns>
        public T GetValue(params int[] indices) => Data[GetFlatIndex(indices)];

        /// <summary>
        /// Устанавливает значение элемента по многомерным индексам.
        /// </summary>
        /// <param name="value">Новое значение элемента.</param>
        /// <param name="indices">Многомерные индексы.</param>
        public void SetValue(T value, params int[] indices) => Data[GetFlatIndex(indices)] = value;

        /// <summary>
        /// Возвращает строковое представление тензора.
        /// </summary>
        /// <returns>Строковое представление.</returns>
        public override string ToString()
        {
            return $"Tensor(shape: [{string.Join(", ", Shape)}], data: [{string.Join(", ", Data)}])";
        }
    }
}