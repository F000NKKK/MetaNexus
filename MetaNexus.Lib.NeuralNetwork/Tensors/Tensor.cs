using MetaNexus.Lib.NeuralNetwork.Tensors.Abstractions;
using Newtonsoft.Json;

namespace MetaNexus.Lib.NeuralNetwork.Tensors
{
    /// <summary>
    /// Структура Tensor представляет многомерный массив числовых данных с поддержкой операций над тензорами.
    /// </summary>
    public partial struct Tensor : ITensor
    {
        private Memory<float> _data;
        private int[] _shape;

        /// <summary>
        /// Конструктор для создания многомерного тензора с заданной формой.
        /// Создает тензор с заданными размерами для каждого измерения.
        /// </summary>
        /// <param name="shape">Массив, представляющий форму тензора (размерности для каждого измерения).</param>
        public Tensor(int[] shape)
        {
            _shape = shape ?? throw new ArgumentNullException(nameof(shape));
            long tempSize = 1L;
            foreach (var dim in shape)
            {
                tempSize *= dim;
                if (tempSize > int.MaxValue) // Проверяем, чтобы размер не превышал int.MaxValue
                {
                    throw new InvalidOperationException("Размер тензора слишком велик для использования в качестве массива.");
                }
            }
            Size = (int)tempSize;
            _data = new Memory<float>(new float[Size]); // Используем Memory вместо массива
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
            _data = new Memory<float>(data ?? throw new ArgumentNullException(nameof(data)));
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
            _data = new Memory<float>(new float[Size]);
            existingTensor._data.Span.CopyTo(_data.Span); // Копирование данных через Span
        }

        public float this[params int[] indices]
        {
            get
            {
                if (indices.Length != _shape.Length)
                    throw new ArgumentException("Количество индексов не соответствует рангу тензора.");

                // Проверяем, что индексы не выходят за пределы каждой размерности
                for (int i = 0; i < indices.Length; i++)
                {
                    if (indices[i] < 0 || indices[i] >= _shape[i])
                        throw new ArgumentException("Индексы выходят за пределы массива.");
                }

                int flatIndex = GetFlatIndex(indices);
                return _data.Span[flatIndex];  // Чтение через Span
            }
            set
            {
                if (indices.Length != _shape.Length)
                    throw new ArgumentException("Количество индексов не соответствует рангу тензора.");

                // Проверяем, что индексы не выходят за пределы каждой размерности
                for (int i = 0; i < indices.Length; i++)
                {
                    if (indices[i] < 0 || indices[i] >= _shape[i])
                        throw new ArgumentException("Индексы выходят за пределы массива.");
                }

                int flatIndex = GetFlatIndex(indices);
                _data.Span[flatIndex] = value;  // Запись через Span
            }
        }

        public float this[int indice]
        {
            get
            {
                return _data.Span[indice];  // Доступ через Span
            }
            set
            {
                _data.Span[indice] = value;  // Запись через Span
            }
        }

        public float[] Data => _data.Span.ToArray();  // Если нужно вернуть данные как обычный массив

        public int[] Shape => _shape;

        public int Rank => _shape.Length;

        public int Size { get; private set; }

        public Tensor Apply(Func<float, float> func)
        {
            var result = new Tensor(_shape);
            for (int i = 0; i < Size; i++)
            {
                result._data.Span[i] = func(_data.Span[i]);
            }
            return result;
        }

        public Tensor Clone()
        {
            var clone = new Tensor(_shape);
            _data.Span.CopyTo(clone._data.Span);  // Копирование через Span
            return clone;
        }

        public float[] FlattenFloatArray()
        {
            return _data.Span.ToArray();  // Если нужно вернуть данные как обычный массив
        }

        public Tensor Flatten()
        {
            // Создаем новый тензор с одномерным массивом
            return new Tensor(new int[] { Size }, _data.Span.ToArray());
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
