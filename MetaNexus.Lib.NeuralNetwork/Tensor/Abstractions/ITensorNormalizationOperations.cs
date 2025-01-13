using System.Numerics;

namespace MetaNexus.Lib.NeuralNetwork.Tensor.Abstractions
{
    /// <summary>
    /// Интерфейс для нормализации тензоров.
    /// </summary>
    internal interface ITensorNormalizationOperations<T> where T : INumber<T>
    {
        /// <summary>
        /// Нормализация значений тензора (например, для подготовки данных).
        /// </summary>
        /// <returns>Нормализованный тензор.</returns>
        Tensor<T> Normalize();

        /// <summary>
        /// Нормализация тензора по батчу.
        /// </summary>
        /// <param name="mean">Среднее значение по батчу.</param>
        /// <param name="variance">Дисперсия по батчу.</param>
        /// <returns>Нормализованный тензор по батчу.</returns>
        Tensor<T> BatchNormalize(Tensor<T> mean, Tensor<T> variance);
    }
}
