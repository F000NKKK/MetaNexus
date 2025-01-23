namespace MetaNexus.Lib.NeuralNetwork.Tensors.Abstractions
{
    /// <summary>
    /// Интерфейс для выполнения матричных операций с тензорами.
    /// </summary>
    public interface ITensorMatrixOperations
    {
        /// <summary>
        /// Матричное умножение двух тензоров.
        /// </summary>
        /// <param name="other">Другой тензор для матричного умножения.</param>
        /// <returns>Результат матричного умножения.</returns>
        Tensor Dot(Tensor other);

        /// <summary>
        /// Матричное деление (аналогично умножению на обратную матрицу).
        /// </summary>
        /// <param name="other">Тензор, на который выполняется деление (или умножение на обратную матрицу).</param>
        /// <returns>Результат матричного деления.</returns>
        Tensor MatrixDivide(Tensor other);

        /// <summary>
        /// Вычисление детерминанта матрицы.
        /// </summary>
        /// <returns>Детерминант матрицы (если применимо).</returns>
        float Determinant();

        /// <summary>
        /// Обращение матрицы.
        /// </summary>
        /// <returns>Обратная матрица.</returns>
        Tensor Inverse();
    }
}
