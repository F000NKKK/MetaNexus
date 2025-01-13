using System.Numerics;

namespace MetaNexus.Lib.NeuralNetwork.Tensor.Abstractions
{
    /// <summary>
    /// Интерфейс для выполнения агрегационных операций с тензорами.
    /// </summary>
    internal interface ITensorAggregationOperations<T> where T : INumber<T>
    {
        /// <summary>
        /// Вычисление суммы всех элементов тензора.
        /// </summary>
        /// <returns>Сумма элементов тензора.</returns>
        T Sum();

        /// <summary>
        /// Нахождение минимального элемента в тензоре.
        /// </summary>
        /// <returns>Минимальное значение в тензоре.</returns>
        T Min();

        /// <summary>
        /// Нахождение максимального элемента в тензоре.
        /// </summary>
        /// <returns>Максимальное значение в тензоре.</returns>
        T Max();

        /// <summary>
        /// Вычисление среднего значения всех элементов тензора.
        /// </summary>
        /// <returns>Среднее значение элементов тензора.</returns>
        T Mean();
    }
}
