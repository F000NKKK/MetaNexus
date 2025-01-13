using System.Numerics;

namespace MetaNexus.Lib.NeuralNetwork.Tensor.Math.Abstractions
{
    /// <summary>
    /// Интерфейс, который объединяет все операции с тензорами.
    /// </summary>
    internal interface ITensorOperations<T>
        : ITensorArithmeticOperations<T>,
          ITensorElementWiseOperations<T>,
          ITensorShapeOperations<T>,
          ITensorAggregationOperations<T>,
          ITensorActivationOperations<T>,
          ITensorNormalizationOperations<T>,
          ITensorMatrixOperations<T>
        where T : INumber<T>
    {
    }
}
