namespace MetaNexus.Lib.NeuralNetwork.Tensors.Abstractions
{
    /// <summary>
    /// Интерфейс для выполнения базовых арифметических операций с тензорами.
    /// </summary>
    internal interface ITensorArithmeticOperations
    {
        /// <summary>
        /// Сложение двух тензоров.
        /// </summary>
        /// <param name="other">Другой тензор для сложения.</param>
        /// <returns>Результат сложения.</returns>
        Tensor Add(Tensor other);

        /// <summary>
        /// Вычитание одного тензора из другого.
        /// </summary>
        /// <param name="other">Тензор, который нужно вычесть.</param>
        /// <returns>Результат вычитания.</returns>
        Tensor Subtract(Tensor other);

        /// <summary>
        /// Умножение двух тензоров.
        /// </summary>
        /// <param name="other">Другой тензор для умножения.</param>
        /// <returns>Результат умножения.</returns>
        Tensor Multiply(Tensor other);

        /// <summary>
        /// Деление одного тензора на другой.
        /// </summary>
        /// <param name="other">Тензор, на который нужно поделить.</param>
        /// <returns>Результат деления.</returns>
        Tensor Divide(Tensor other);

        /// <summary>
        /// Сложение тензора и скаляра.
        /// </summary>
        /// <param name="scalar">Скаляр, который нужно добавить к каждому элементу тензора.</param>
        /// <returns>Результат сложения с числом.</returns>
        Tensor Add(float scalar);

        /// <summary>
        /// Вычитание скаляра из тензора.
        /// </summary>
        /// <param name="scalar">Скаляр, который нужно вычесть из каждого элемента тензора.</param>
        /// <returns>Результат вычитания скаляра.</returns>
        Tensor Subtract(float scalar);

        /// <summary>
        /// Умножение тензора на скаляр.
        /// </summary>
        /// <param name="scalar">Скаляр, на который нужно умножить каждый элемент тензора.</param>
        /// <returns>Результат умножения с числом.</returns>
        Tensor Multiply(float scalar);

        /// <summary>
        /// Деление тензора на скаляр.
        /// </summary>
        /// <param name="scalar">Скаляр, на который нужно поделить каждый элемент тензора.</param>
        /// <returns>Результат деления на число.</returns>
        Tensor Divide(float scalar);
    }
}
