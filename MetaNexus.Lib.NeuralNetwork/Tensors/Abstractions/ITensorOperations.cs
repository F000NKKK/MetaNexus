namespace MetaNexus.Lib.NeuralNetwork.Tensors.Abstractions
{
    /// <summary>
    /// Интерфейс, который объединяет все операции с тензорами.
    /// </summary>
    public interface ITensorOperations
        : ITensorArithmeticOperations,
          ITensorElementWiseOperations,
          ITensorShapeOperations,
          ITensorAggregationOperations,
          ITensorActivationOperations,
          ITensorActivationOperationsPrime,
          ITensorNormalizationOperations,
          ITensorMatrixOperations
    {
    }
}
