namespace MetaNexus.Lib.NeuralNetwork.Tensor.Abstractions
{
    /// <summary>
    /// Интерфейс для нормализации тензоров.
    /// </summary>
    internal interface ITensorNormalizationOperations
    {
        /// <summary>
        /// Нормализация значений тензора (например, для подготовки данных).
        /// </summary>
        /// <returns>Нормализованный тензор.</returns>
        Tensor Normalize();

        /// <summary>
        /// Нормализация тензора по батчу.
        /// </summary>
        /// <param name="mean">Среднее значение по батчу.</param>
        /// <param name="variance">Дисперсия по батчу.</param>
        /// <returns>Нормализованный тензор по батчу.</returns>
        Tensor BatchNormalize(Tensor mean, Tensor variance);
    }
}
