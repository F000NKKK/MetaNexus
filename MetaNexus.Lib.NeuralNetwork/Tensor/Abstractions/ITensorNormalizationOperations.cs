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

        /// <summary>
        /// Нормализация значений тензора в пределах диапазона [0, 1] (Min-Max нормализация).
        /// </summary>
        /// <returns>Нормализованный тензор в диапазоне [0, 1].</returns>
        Tensor MinMaxNormalize();

        /// <summary>
        /// Нормализация по каналу, используемая для сверток или работы с многоканальными данными.
        /// </summary>
        /// <param name="mean">Среднее значение по каналу.</param>
        /// <param name="variance">Дисперсия по каналу.</param>
        /// <returns>Нормализованный тензор по каналу.</returns>
        Tensor ChannelNormalize(Tensor mean, Tensor variance);

        /// <summary>
        /// Стандартизация значений тензора (централизованная нормализация, с вычитанием среднего и делением на стандартное отклонение).
        /// </summary>
        /// <returns>Стандартизованный тензор.</returns>
        Tensor Standardize();

        /// <summary>
        /// Лейбловая нормализация, используемая для подготовки данных для классификации.
        /// </summary>
        /// <param name="numClasses">Количество классов в задаче классификации.</param>
        /// <returns>Нормализованный тензор для задачи классификации.</returns>
        Tensor LabelNormalize(int numClasses);
    }
}
