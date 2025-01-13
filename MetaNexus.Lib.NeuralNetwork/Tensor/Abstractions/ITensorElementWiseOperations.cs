namespace MetaNexus.Lib.NeuralNetwork.Tensor.Abstractions
{
    /// <summary>
    /// Интерфейс для выполнения поэлементных операций с тензорами.
    /// Позволяет реализовать операции, такие как сложение, вычитание, умножение и деление,
    /// между тензорами или между тензором и скаляром.
    /// </summary>
    internal interface ITensorElementWiseOperations
    {
        /// <summary>
        /// Выполняет поэлементную операцию между текущим тензором и другим тензором.
        /// Ожидается, что формы (размерности) обоих тензоров совпадают.
        /// </summary>
        /// <param name="other">Другой тензор, с которым будет выполнена операция.</param>
        /// <param name="operation">Функция, описывающая операцию, которая будет применена к каждому элементу.</param>
        /// <returns>Новый тензор, результат выполнения операции.</returns>
        Tensor ElementWiseOperation(Tensor other, Func<float, float, float> operation);

        /// <summary>
        /// Выполняет поэлементную операцию между текущим тензором и скаляром.
        /// Скаляры могут быть использованы для масштабирования или смещения значений в тензоре.
        /// </summary>
        /// <param name="scalar">Скаляр, с которым будет выполнена операция.</param>
        /// <param name="operation">Функция, описывающая операцию, которая будет применена к каждому элементу.</param>
        /// <returns>Новый тензор, результат выполнения операции.</returns>
        Tensor ElementWiseOperation(float scalar, Func<float, float, float> operation);
    }
}
