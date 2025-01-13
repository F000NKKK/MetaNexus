namespace MetaNexus.Lib.NeuralNetwork.Tensor.Abstractions
{
    /// <summary>
    /// Интерфейс для выполнения матричных операций с тензорами.
    /// </summary>
    internal interface ITensorMatrixOperations
    {
        /// <summary>
        /// Матричное умножение двух тензоров.
        /// </summary>
        /// <param name="other">Другой тензор для матричного умножения.</param>
        /// <returns>Результат матричного умножения.</returns>
        Tensor MatrixMultiply(Tensor other);
    }
}
