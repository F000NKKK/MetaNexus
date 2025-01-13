using MetaNexus.Lib.NeuralNetwork.Tensors.Abstractions;

namespace MetaNexus.Lib.NeuralNetwork.Tensors
{
    public partial struct Tensor : ITensorElementWiseOperations
    {
        Tensor ITensorElementWiseOperations.ElementWiseOperation(Tensor other, Func<float, float, float> operation)
        {
            if (!Shape.SequenceEqual(other.Shape))
                throw new InvalidOperationException("Tensors must have the same shape for element-wise operation.");

            Tensor result = new Tensor(Shape);
            for (int i = 0; i < Size; i++)
            {
                result._data[i] = operation(_data[i], other._data[i]);
            }
            return result;
        }

        Tensor ITensorElementWiseOperations.ElementWiseOperation(float scalar, Func<float, float, float> operation)
        {
            Tensor result = new Tensor(Shape);
            for (int i = 0; i < Size; i++)
            {
                result._data[i] = operation(_data[i], scalar);
            }
            return result;
        }
    }
}
