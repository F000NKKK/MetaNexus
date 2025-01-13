namespace MetaNexus.Lib.NeuralNetwork.Tensor.Abstractions
{
    /// <summary>
    /// Интерфейс для выполнения поэлементных операций с тензорами.
    /// </summary>
    internal interface ITensorElementWiseOperations
    {
        /// <summary>
        /// Поэлементное сложение двух тензоров.
        /// </summary>
        /// <param name="other">Другой тензор для поэлементного сложения.</param>
        /// <returns>Результат поэлементного сложения.</returns>
        Tensor ElementWiseAdd(Tensor other);

        /// <summary>
        /// Поэлементное вычитание двух тензоров.
        /// </summary>
        /// <param name="other">Тензор для поэлементного вычитания.</param>
        /// <returns>Результат поэлементного вычитания.</returns>
        Tensor ElementWiseSubtract(Tensor other);

        /// <summary>
        /// Поэлементное умножение двух тензоров.
        /// </summary>
        /// <param name="other">Тензор для поэлементного умножения.</param>
        /// <returns>Результат поэлементного умножения.</returns>
        Tensor ElementWiseMultiply(Tensor other);

        /// <summary>
        /// Поэлементное деление двух тензоров.
        /// </summary>
        /// <param name="other">Тензор для поэлементного деления.</param>
        /// <returns>Результат поэлементного деления.</returns>
        Tensor ElementWiseDivide(Tensor other);
    }
}
