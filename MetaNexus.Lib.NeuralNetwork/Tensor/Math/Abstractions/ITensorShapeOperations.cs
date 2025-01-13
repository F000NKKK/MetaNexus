using MetaNexus.Lib.NeuralNetwork.Math.Tensor;
using System.Numerics;

namespace MetaNexus.Lib.NeuralNetwork.Tensor.Math.Abstractions
{
    /// <summary>
    /// Интерфейс для изменения формы и трансформации тензоров.
    /// </summary>
    internal interface ITensorShapeOperations<T> where T : INumber<T>
    {
        /// <summary>
        /// Транспонирование тензора (для матриц).
        /// </summary>
        /// <returns>Тензор, полученный в результате транспонирования.</returns>
        Tensor<T> Transpose();

        /// <summary>
        /// Изменение формы тензора.
        /// </summary>
        /// <param name="newShape">Новая форма тензора.</param>
        /// <returns>Тензор с новой формой.</returns>
        Tensor<T> Reshape(int[] newShape);
    }
}
