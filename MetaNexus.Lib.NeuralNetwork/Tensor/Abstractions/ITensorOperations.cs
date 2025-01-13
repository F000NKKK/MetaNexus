namespace MetaNexus.Lib.NeuralNetwork.Tensor.Abstractions
{
    /// <summary>
    /// Интерфейс, который объединяет все операции с тензорами.
    /// </summary>
    internal interface ITensorOperations
        : ITensorArithmeticOperations,
          ITensorElementWiseOperations,
          ITensorShapeOperations,
          ITensorAggregationOperations,
          ITensorActivationOperations,
          ITensorNormalizationOperations,
          ITensorMatrixOperations
    {
    }
}
