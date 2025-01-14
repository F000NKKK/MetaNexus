using MetaNexus.Lib.NeuralNetwork.Tensors.Abstractions;

namespace MetaNexus.Lib.NeuralNetwork.Tensors
{
    public partial struct Tensor : ITensorAggregationOperations
    {
        public float Max()
        {
            if (_data == null || _data.Length == 0)
                throw new InvalidOperationException("Тензор пуст");

            return _data.Max();
        }

        public float Min()
        {
            if (_data == null || _data.Length == 0)
                throw new InvalidOperationException("Тензор пуст");

            return _data.Min();
        }

        public float Avg()
        {
            if (_data == null || _data.Length == 0)
                throw new InvalidOperationException("Тензор пуст");

            return _data.Average();
        }

        public float Sum()
        {
            if (_data == null || _data.Length == 0)
                throw new InvalidOperationException("Тензор пуст");

            return _data.Sum();
        }
        public Tensor Sum(int axis)
        {
            if (axis < 0 || axis >= Rank)
                throw new ArgumentOutOfRangeException(nameof(axis), "Недопустимая ось для операции Sum.");

            int[] resultShape = Shape.ToArray();
            resultShape[axis] = 1;
            Tensor result = new Tensor(resultShape);

            int[] indices = new int[Rank];
            for (int i = 0; i < Size; i++)
            {
                int[] originalIndices = GetIndicesFromFlatIndex(i);
                indices[axis] = 0;
                for (int j = 0; j < Shape[axis]; j++)
                {
                    indices[axis] = j;
                    result[originalIndices] += this[indices];
                }
            }
            return result;
        }
    }
}
