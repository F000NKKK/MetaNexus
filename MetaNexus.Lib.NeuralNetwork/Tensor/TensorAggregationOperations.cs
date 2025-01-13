using MetaNexus.Lib.NeuralNetwork.Tensor.Abstractions;

namespace MetaNexus.Lib.NeuralNetwork.Tensor
{
    public partial struct Tensor : ITensorAggregationOperations
    {

        // Нахождение максимального элемента в тензоре
        public float Max()
        {
            if (_data == null || _data.Length == 0)
                throw new InvalidOperationException("Тензор пуст");

            return _data.Max();
        }

        // Нахождение минимального элемента в тензоре
        public float Min()
        {
            if (_data == null || _data.Length == 0)
                throw new InvalidOperationException("Тензор пуст");

            return _data.Min();
        }

        // Вычисление среднего значения всех элементов тензора
        public float Avg()
        {
            if (_data == null || _data.Length == 0)
                throw new InvalidOperationException("Тензор пуст");

            return _data.Average();
        }

        // Вычисление суммы всех элементов тензора
        public float Sum()
        {
            if (_data == null || _data.Length == 0)
                throw new InvalidOperationException("Тензор пуст");

            return _data.Sum();
        }
    }
}
