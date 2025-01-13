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

        /// <summary>
        /// Разбиение тензора на несколько частей вдоль указанной оси.
        /// </summary>
        /// <param name="axis">Ось, вдоль которой будет произведено разбиение.</param>
        /// <param name="parts">Количество частей для разбиения.</param>
        /// <returns>Массив тензоров, полученных после разбиения.</returns>
        Tensor[] Split(int axis, int parts);

        /// <summary>
        /// Изменение порядка осей тензора.
        /// </summary>
        /// <param name="newOrder">Новый порядок осей.</param>
        /// <returns>Тензор с изменённым порядком осей.</returns>
        Tensor TransposeAxes(int[] newOrder);

    }
}
