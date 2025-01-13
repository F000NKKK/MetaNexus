namespace MetaNexus.Lib.NeuralNetwork.Tensor.Abstractions
{
    /// <summary>
    /// Интерфейс для выполнения агрегационных операций с тензорами.
    /// </summary>
    internal interface ITensorAggregationOperations
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
    }
}
