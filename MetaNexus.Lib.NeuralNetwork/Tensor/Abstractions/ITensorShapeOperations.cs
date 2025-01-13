namespace MetaNexus.Lib.NeuralNetwork.Tensor.Abstractions
{
    /// <summary>
    /// Интерфейс для изменения формы и трансформации тензоров.
    /// </summary>
    internal interface ITensorShapeOperations
    {
        /// <summary>
        /// Транспонирование тензора (для матриц).
        /// </summary>
        /// <returns>Тензор, полученный в результате транспонирования.</returns>
        Tensor Transpose();

        /// <summary>
        /// Изменение формы тензора.
        /// </summary>
        /// <param name="newShape">Новая форма тензора.</param>
        /// <returns>Тензор с новой формой.</returns>
        Tensor Reshape(int[] newShape);
    }
}
