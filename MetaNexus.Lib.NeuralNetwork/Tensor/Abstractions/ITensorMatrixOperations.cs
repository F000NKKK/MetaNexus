using System.Numerics;

namespace MetaNexus.Lib.NeuralNetwork.Tensor.Abstractions
{
    /// <summary>
    /// Интерфейс для выполнения матричных операций с тензорами.
    /// </summary>
    internal interface ITensorMatrixOperations<T> where T : INumber<T>
    {
        /// <summary>
        /// Матричное умножение двух тензоров.
        /// </summary>
        /// <param name="other">Другой тензор для матричного умножения.</param>
        /// <returns>Результат матричного умножения.</returns>
        Tensor<T> MatrixMultiply(Tensor<T> other);
    }
}
