using System.Numerics;

namespace MetaNexus.Lib.NeuralNetwork.Tensor.Abstractions
{
    /// <summary>
    /// Интерфейс для выполнения базовых арифметических операций с тензорами.
    /// </summary>
    internal interface ITensorArithmeticOperations<T> where T : INumber<T>
    {
        /// <summary>
        /// Сложение двух тензоров.
        /// </summary>
        /// <param name="other">Другой тензор для сложения.</param>
        /// <returns>Результат сложения.</returns>
        Tensor<T> Add(Tensor<T> other);

        /// <summary>
        /// Вычитание одного тензора из другого.
        /// </summary>
        /// <param name="other">Тензор, который нужно вычесть.</param>
        /// <returns>Результат вычитания.</returns>
        Tensor<T> Subtract(Tensor<T> other);

        /// <summary>
        /// Умножение двух тензоров.
        /// </summary>
        /// <param name="other">Другой тензор для умножения.</param>
        /// <returns>Результат умножения.</returns>
        Tensor<T> Multiply(Tensor<T> other);

        /// <summary>
        /// Деление одного тензора на другой.
        /// </summary>
        /// <param name="other">Тензор, на который нужно поделить.</param>
        /// <returns>Результат деления.</returns>
        Tensor<T> Divide(Tensor<T> other);

        /// <summary>
        /// Сложение тензора и скаляра.
        /// </summary>
        /// <param name="scalar">Скаляр, который нужно добавить к каждому элементу тензора.</param>
        /// <returns>Результат сложения с числом.</returns>
        Tensor<T> Add(T scalar);

        /// <summary>
        /// Вычитание скаляра из тензора.
        /// </summary>
        /// <param name="scalar">Скаляр, который нужно вычесть из каждого элемента тензора.</param>
        /// <returns>Результат вычитания скаляра.</returns>
        Tensor<T> Subtract(T scalar);

        /// <summary>
        /// Умножение тензора на скаляр.
        /// </summary>
        /// <param name="scalar">Скаляр, на который нужно умножить каждый элемент тензора.</param>
        /// <returns>Результат умножения с числом.</returns>
        Tensor<T> Multiply(T scalar);

        /// <summary>
        /// Деление тензора на скаляр.
        /// </summary>
        /// <param name="scalar">Скаляр, на который нужно поделить каждый элемент тензора.</param>
        /// <returns>Результат деления на число.</returns>
        Tensor<T> Divide(T scalar);
    }
}
