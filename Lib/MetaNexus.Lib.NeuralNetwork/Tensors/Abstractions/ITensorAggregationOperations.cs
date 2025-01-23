namespace MetaNexus.Lib.NeuralNetwork.Tensors.Abstractions
{
    /// <summary>
    /// Интерфейс для выполнения агрегационных операций с тензорами.
    /// </summary>
    public interface ITensorAggregationOperations
    {
        /// <summary>
        /// Вычисление суммы всех элементов тензора.
        /// </summary>
        /// <returns>Сумма элементов тензора.</returns>
        float Sum();

        /// <summary>
        /// Нахождение минимального элемента в тензоре.
        /// </summary>
        /// <returns>Минимальное значение в тензоре.</returns>
        float Min();

        /// <summary>
        /// Нахождение максимального элемента в тензоре.
        /// </summary>
        /// <returns>Максимальное значение в тензоре.</returns>
        float Max();

        /// <summary>
        /// Вычисление среднего значения всех элементов тензора.
        /// </summary>
        /// <returns>Среднее значение элементов тензора.</returns>
        float Avg();

        /// <summary>
        /// Вычисление среднего элементов вдоль указанной оси.
        /// </summary>
        /// <param name="axis">Ось, вдоль которой будет вычислено среднее.</param>
        /// <returns></returns>
        Tensor Avg(int axis);

        /// <summary>
        /// Вычисление суммы элементов вдоль указанной оси.
        /// </summary>
        /// <param name="axis">Ось, вдоль которой будет вычислена сумма.</param>
        /// <returns>Новый тензор с результатами суммирования.</returns>
        /// <exception cref="ArgumentOutOfRangeException">Бросается, если ось выходит за пределы допустимых значений.</exception>
        Tensor Sum(int axis);

        /// <summary>
        /// Вычисление дисперсии элементов вдоль указанной оси.
        /// </summary>
        /// <param name="axis">Ось, вдоль которой будет вычислена дисперсия.</param>
        /// <returns>Новый тензор с результатами вычисления дисперсии.</returns>
        /// <exception cref="ArgumentOutOfRangeException">Бросается, если ось выходит за пределы допустимых значений.</exception>
        Tensor Variance(int axis);
    }
}
