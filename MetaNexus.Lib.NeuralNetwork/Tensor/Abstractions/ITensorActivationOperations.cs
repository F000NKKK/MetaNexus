using System.Numerics;

namespace MetaNexus.Lib.NeuralNetwork.Tensor.Abstractions
{
    /// <summary>
    /// Интерфейс для применения функций активации к тензорам.
    /// </summary>
    internal interface ITensorActivationOperations<T> where T : INumber<T>
    {
        /// <summary>
        /// Применение функции активации ReLU к тензору.
        /// </summary>
        /// <returns>Результат применения ReLU.</returns>
        Tensor<T> ApplyReLU();

        /// <summary>
        /// Применение функции активации сигмоид к тензору.
        /// </summary>
        /// <returns>Результат применения сигмоиды.</returns>
        Tensor<T> ApplySigmoid();

        /// <summary>
        /// Применение функции активации tanh (гиперболический тангенс) к тензору.
        /// </summary>
        /// <returns>Результат применения tanh.</returns>
        Tensor<T> ApplyTanh();
    }
}
