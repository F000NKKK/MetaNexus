using MetaNexus.Lib.NeuralNetwork.Math.Tensor;
using System.Numerics;

namespace MetaNexus.Lib.NeuralNetwork.Tensor.Math.Abstractions
{
    /// <summary>
    /// Интерфейс для выполнения поэлементных операций с тензорами.
    /// </summary>
    internal interface ITensorElementWiseOperations<T> where T : INumber<T>
    {
        /// <summary>
        /// Поэлементное сложение двух тензоров.
        /// </summary>
        /// <param name="other">Другой тензор для поэлементного сложения.</param>
        /// <returns>Результат поэлементного сложения.</returns>
        Tensor<T> ElementWiseAdd(Tensor<T> other);

        /// <summary>
        /// Поэлементное вычитание двух тензоров.
        /// </summary>
        /// <param name="other">Тензор для поэлементного вычитания.</param>
        /// <returns>Результат поэлементного вычитания.</returns>
        Tensor<T> ElementWiseSubtract(Tensor<T> other);

        /// <summary>
        /// Поэлементное умножение двух тензоров.
        /// </summary>
        /// <param name="other">Тензор для поэлементного умножения.</param>
        /// <returns>Результат поэлементного умножения.</returns>
        Tensor<T> ElementWiseMultiply(Tensor<T> other);

        /// <summary>
        /// Поэлементное деление двух тензоров.
        /// </summary>
        /// <param name="other">Тензор для поэлементного деления.</param>
        /// <returns>Результат поэлементного деления.</returns>
        Tensor<T> ElementWiseDivide(Tensor<T> other);
    }
}
