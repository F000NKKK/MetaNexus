using MetaNexus.Lib.NeuralNetwork.Tensors.Abstractions;

namespace MetaNexus.Lib.NeuralNetwork.Tensors
{
    public partial struct Tensor : ITensorElementWiseOperations
    {
        public Tensor ElementWiseOperation(ITensor other, Func<float, float, float> operation)
        {
            if (!Shape.SequenceEqual(other.Shape))
                throw new InvalidOperationException("Tensors must have the same shape for element-wise operation.");

            Tensor result = new Tensor(Shape);
            for (int i = 0; i < Size; i++)
            {
                result._data.Span[i] = operation(_data.Span[i], other.Data[i]);
            }
            return result;
        }


        public Tensor ElementWiseOperation(float scalar, Func<float, float, float> operation)
        {
            Tensor result = new Tensor(Shape);
            for (int i = 0; i < Size; i++)
            {
                result._data.Span[i] = operation(_data.Span[i], scalar);
            }
            return result;
        }
    }
}
