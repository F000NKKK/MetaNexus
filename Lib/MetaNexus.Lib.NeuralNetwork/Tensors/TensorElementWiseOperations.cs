using MetaNexus.Lib.NeuralNetwork.Tensors.Abstractions;

namespace MetaNexus.Lib.NeuralNetwork.Tensors
{
    public partial struct Tensor : ITensorElementWiseOperations
    {
        public Tensor ElementWiseOperation(ITensor other, Func<float, float, float> operation)
        {
            if (!Shape.SequenceEqual(other.Shape))
                throw new InvalidOperationException("Тензоры должны иметь одинаковую форму для поэлементной работы.");

            Tensor result = new Tensor(Shape);
            for (int i = 0; i < Size; i++)
            {
                result._data.Span[i] = operation(_data.Span[i], other.Data.Span[i]);
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

        public Tensor ElementWiseOperation(Func<float, float> operation)
        {
            Tensor result = new Tensor(Shape);
            for (int i = 0; i < Size; i++)
            {
                result._data.Span[i] = operation(_data.Span[i]);
            }
            return result;
        }

    }
}
