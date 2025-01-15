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
            {
                throw new ArgumentOutOfRangeException(nameof(axis), "Ось для агрегации выходит за пределы ранга тензора.");
            }

            // Создаем форму для результирующего тензора
            int[] resultShape = (int[])Shape.Clone();
            resultShape[axis] = 1;  // Размерность по оси уменьшается до 1 после агрегации

            // Создаем новый тензор для хранения результата
            var result = new Tensor(resultShape);
            int[] indices = new int[Rank];  // Индексы для исходного тензора

            // Перебираем все индексы результирующего тензора
            for (int i = 0; i < result.Size; i++)
            {
                // Преобразуем плоский индекс в многомерный для результирующего тензора
                int[] resultIndices = new int[Rank];
                int tempIndex = i;
                for (int d = Rank - 1; d >= 0; d--)
                {
                    resultIndices[d] = tempIndex % resultShape[d];
                    tempIndex /= resultShape[d];
                }

                float sum = 0;

                // Для каждой позиции по оси агрегации суммируем элементы
                for (int j = 0; j < Shape[axis]; j++)
                {
                    // Копируем индексы для исходного тензора и изменяем только индекс по оси агрегации
                    Array.Copy(resultIndices, indices, Rank);
                    indices[axis] = j;  // Изменяем только по оси агрегации

                    // Добавляем значение из исходного тензора в сумму
                    sum += this[indices];
                }

                // Записываем сумму в результат
                result[resultIndices] = sum;
            }

            return result;
        }


    }
}
