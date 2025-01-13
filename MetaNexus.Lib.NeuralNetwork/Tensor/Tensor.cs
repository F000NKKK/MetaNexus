using MetaNexus.Lib.NeuralNetwork.Tensor.Abstractions;
using System.Numerics;

namespace MetaNexus.Lib.NeuralNetwork.Tensor
{
    /// <summary>
    /// Структура Tensor представляет многомерный массив числовых данных с поддержкой операций над тензорами.
    /// </summary>
    public partial struct Tensor<T> : ITensor<T> where T : INumber<T>
    {
        private T[] _data;
        private int[] _shape;

        // Конструктор для создания тензора из данных и формы
        public Tensor(int[] shape)
        {
            _shape = shape ?? throw new ArgumentNullException(nameof(shape));
            Size = 1;
            foreach (var dim in shape)
            {
                Size *= dim;
            }
            _data = new T[Size];
        }

        public T this[params int[] indices]
        {
            get
            {
                if (indices.Length != _shape.Length)
                    throw new ArgumentException("Количество индексов не соответствует рангу тензора.");

                int flatIndex = GetFlatIndex(indices);
                return _data[flatIndex];
            }
            set
            {
                if (indices.Length != _shape.Length)
                    throw new ArgumentException("Количество индексов не соответствует рангу тензора.");

                int flatIndex = GetFlatIndex(indices);
                _data[flatIndex] = value;
            }
        }

        public int[] Shape => _shape;

        public int Rank => _shape.Length;

        public int Size { get; private set; }

        public Tensor<T> Apply(Func<T, T> func)
        {
            var result = new Tensor<T>(_shape);
            for (int i = 0; i < Size; i++)
            {
                result._data[i] = func(_data[i]);
            }
            return result;
        }

        public Tensor<T> Clone()
        {
            var clone = new Tensor<T>(_shape);
            Array.Copy(_data, clone._data, Size);
            return clone;
        }

        public Tensor<TTarget> ConvertTo<TTarget>() where TTarget : INumber<TTarget>
        {
            var targetTensor = new Tensor<TTarget>(_shape);
            for (int i = 0; i < Size; i++)
            {
                targetTensor._data[i] = (TTarget)Convert.ChangeType(_data[i], typeof(TTarget));
            }
            return targetTensor;
        }

        public T[] Flatten()
        {
            return (T[])_data.Clone();
        }

        public bool IsEmpty()
        {
            return Size == 0;
        }

        private int GetFlatIndex(int[] indices)
        {
            int index = 0;
            int multiplier = 1;
            for (int i = _shape.Length - 1; i >= 0; i--)
            {
                index += indices[i] * multiplier;
                multiplier *= _shape[i];
            }
            return index;
        }
    }
}
