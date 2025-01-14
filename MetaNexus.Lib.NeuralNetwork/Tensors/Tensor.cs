using MetaNexus.Lib.NeuralNetwork.Tensors.Abstractions;
using Newtonsoft.Json;

namespace MetaNexus.Lib.NeuralNetwork.Tensors
{
    /// <summary>
    /// Структура Tensor представляет многомерный массив числовых данных с поддержкой операций над тензорами.
    /// </summary>
    public partial struct Tensor : ITensor
    {
        private float[] _data;
        private int[] _shape;

        /// <summary>
        /// Конструктор для создания многомерного тензора с заданной формой.
        /// Создает тензор с заданными размерами для каждого измерения.
        /// </summary>
        /// <param name="shape">Массив, представляющий форму тензора (размерности для каждого измерения).</param>
        public Tensor(int[] shape)
        {
            _shape = shape ?? throw new ArgumentNullException(nameof(shape));
            Size = 1;
            foreach (var dim in shape)
            {
                Size *= dim;
            }
            _data = new float[Size];
        }

        /// <summary>
        /// Конструктор для создания многомерного тензора с заданной формой и исходными данными.
        /// </summary>
        /// <param name="shape">Массив, представляющий форму тензора (размерности для каждого измерения).</param>
        /// <param name="data">Массив данных, заполняющий тензор.</param>
        public Tensor(int[] shape, float[] data)
        {
            _shape = shape ?? throw new ArgumentNullException(nameof(shape));
            Size = 1;
            foreach (var dim in shape)
            {
                Size *= dim;
            }
            _data = data ?? throw new ArgumentNullException(nameof(data));
            if (data.Length != Size)
            {
                throw new ArgumentException("Размер массива данных не соответствует размеру тензора.");
            }
        }

        /// <summary>
        /// Конструктор для создания нового тензора на основе существующего тензора (глубокое копирование).
        /// </summary>
        /// <param name="existingTensor">Существующий тензор, данные которого будут скопированы в новый.</param>
        public Tensor(Tensor existingTensor)
        {
            _shape = existingTensor._shape;
            Size = existingTensor.Size;
            _data = new float[Size];
            Array.Copy(existingTensor._data, _data, Size);
        }

        public float this[params int[] indices]
        {
            get
            {
                if (indices.Length != _shape.Length)
                    throw new ArgumentException("Количество индексов не соответствует рангу тензора.");

                int flatIndex = GetFlatIndex(indices);
                return this[flatIndex];
            }
            set
            {
                if (indices.Length != _shape.Length)
                    throw new ArgumentException("Количество индексов не соответствует рангу тензора.");

                int flatIndex = GetFlatIndex(indices);
                this[flatIndex] = value;
            }
        }

        public float this[int indice]
        {
            get
            {
                return _data[indice];
            }
            set
            {
                _data[indice] = value;
            }
        }

        public float[] Data => _data;
        public int[] Shape => _shape;

        public int Rank => _shape.Length;

        public int Size { get; private set; }

        public Tensor Apply(Func<float, float> func)
        {
            var result = new Tensor(_shape);
            for (int i = 0; i < Size; i++)
            {
                result._data[i] = func(_data[i]);
            }
            return result;
        }

        public Tensor Clone()
        {
            var clone = new Tensor(_shape);
            Array.Copy(_data, clone._data, Size);
            return clone;
        }

        public float[] FlattenFloatArray()
        {
            return (float[])_data.Clone();
        }
        public Tensor Flatten()
        {
            // Создаем новый тензор с одномерным массивом
            return new Tensor(new int[] { Size }, _data);
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

        private int[] GetIndicesFromFlatIndex(int flatIndex)
        {
            int[] indices = new int[Rank];
            for (int i = Rank - 1; i >= 0; i--)
            {
                indices[i] = flatIndex % Shape[i];
                flatIndex /= Shape[i];
            }
            return indices;
        }

        private int GetFlatIndexFromNewOrder(int[] indices, int[] newOrder)
        {
            int flatIndex = 0;
            int multiplier = 1;

            for (int i = _shape.Length - 1; i >= 0; i--)
            {
                flatIndex += indices[newOrder[i]] * multiplier;
                multiplier *= _shape[newOrder[i]];
            }

            return flatIndex;
        }

        public override string ToString()
        {
            return JsonConvert.SerializeObject(this, Formatting.Indented);
        }
    }
}
