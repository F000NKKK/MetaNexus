using MetaNexus.Lib.NeuralNetwork.Tensors.Abstractions;

namespace MetaNexus.Lib.NeuralNetwork.Tensors
{
    public partial struct Tensor : ITensorAggregationOperations
    {
        public float Max()
        {
            if (_data.Length == 0)
                throw new InvalidOperationException("Тензор пуст");

            return _data.Span.ToArray().Max();
        }

        public float Min()
        {
            if (_data.Length == 0)
                throw new InvalidOperationException("Тензор пуст");

            return _data.Span.ToArray().Min();
        }

        public float Avg()
        {
            if (_data.Length == 0)
                throw new InvalidOperationException("Тензор пуст");

            return _data.Span.ToArray().Average();
        }

        public float Sum()
        {
            if (_data.Length == 0)
                throw new InvalidOperationException("Тензор пуст");

            return _data.Span.ToArray().Sum();
        }

        public Tensor Sum(int axis)
        {
            if (axis < 0 || axis >= Rank)
                throw new ArgumentOutOfRangeException(nameof(axis), "Недопустимая ось для операции Sum.");

            int[] resultShape = Shape.ToArray();
            resultShape[axis] = 1; // Мы хотим агрегировать по оси, поэтому уменьшаем размерность

            Tensor result = new Tensor(resultShape);  // Создаем новый тензор для результата
            int[] indices = new int[Rank];  // Индексы для всех измерений

            for (int i = 0; i < Size; i++)
            {
                // Получаем индексы для текущего элемента в тензоре
                int[] originalIndices = GetIndicesFromFlatIndex(i);

                // Сбросим индекс по оси для агрегирования
                indices[axis] = 0;

                // Пройдем по всем возможным индексам для этой оси
                for (int j = 0; j < Shape[axis]; j++)
                {
                    // Для каждого индекса по оси суммируем элементы
                    indices[axis] = j;
                    result[originalIndices] += this[indices];
                }
            }
            return result;
        }
    }
}
