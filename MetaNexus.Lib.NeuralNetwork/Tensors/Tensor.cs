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

        public Tensor(int[] shape)
        {
            _shape = shape ?? throw new ArgumentNullException(nameof(shape));

            long tempSize = 1L;
            foreach (var dim in shape)
            {
                tempSize *= dim;
                if (tempSize > int.MaxValue)  // Проверка на слишком большой размер
                {
                    throw new InvalidOperationException("Размер тензора слишком велик для использования в качестве массива.");
                }
            }

            Size = (int)tempSize;
            _data = new Memory<float>(new float[Size]);
        }

        public Tensor(int[] shape, float[] data)
        {
            _shape = shape ?? throw new ArgumentNullException(nameof(shape));

            Size = 1;
            foreach (var dim in shape)
            {
                Size *= dim;
            }

            if (data == null) throw new ArgumentNullException(nameof(data));
            if (data.Length != Size)
            {
                throw new ArgumentException("Размер массива данных не соответствует размеру тензора.");
            }

            _data = new Memory<float>(data);
        }

        public Tensor(int[] shape, Memory<float> data)
        {
            _shape = shape ?? throw new ArgumentNullException(nameof(shape));

            Size = 1;
            foreach (var dim in shape)
            {
                Size *= dim;
            }

            if (data.IsEmpty) throw new ArgumentNullException(nameof(data));
            if (data.Length != Size)
            {
                throw new ArgumentException("Размер массива данных не соответствует размеру тензора.");
            }

            _data = data;
        }

        public Tensor(Tensor existingTensor)
        {
            _shape = existingTensor._shape;
            Size = existingTensor.Size;
            _data = new Memory<float>(new float[Size]);
            existingTensor._data.Span.CopyTo(_data.Span);
        }

        public float this[params int[] indices]
        {
            get
            {
                if (indices.Length != _shape.Length)
                    throw new ArgumentException("Количество индексов не соответствует рангу тензора.");

                for (int i = 0; i < indices.Length; i++)
                {
                    if (indices[i] < 0 || indices[i] >= _shape[i])
                        throw new ArgumentException("Индексы выходят за пределы массива.");
                }

                int flatIndex = GetFlatIndex(indices);
                return _data.Span[flatIndex];  // Доступ через Span
            }
            set
            {
                if (indices.Length != _shape.Length)
                    throw new ArgumentException("Количество индексов не соответствует рангу тензора.");


                for (int i = 0; i < indices.Length; i++)
                {
                    if (indices[i] < 0 || indices[i] >= _shape[i])
                        throw new ArgumentException("Индексы выходят за пределы массива.");
                }

                int flatIndex = GetFlatIndex(indices);
                _data.Span[flatIndex] = value;
            }
        }

        private int GetFlatIndex(int[] indices)
        {
            int flatIndex = 0;
            for (int i = 0; i < indices.Length; i++)
            {
                int stride = 1;
                for (int j = i + 1; j < indices.Length; j++)
                {
                    stride *= _shape[j];
                }
                flatIndex += indices[i] * stride;
            }
            return flatIndex;
        }

        private float this[int indice]
        {
            get
            {
                return _data.Span[indice];
            }
            set
            {
                _data.Span[indice] = value;
            }
        }

        public float[] Data => _data.Span.ToArray();

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

        /// <summary>
        /// Преобразует плоский индекс в многомерный индекс, соответствующий форме тензора.
        /// </summary>
        /// <param name="flatIndex">Плоский индекс (индекс элемента в одномерном массиве, который соответствует многомерному тензору).</param>
        /// <returns>Массив индексов, соответствующих многомерному положению элемента.</returns>
        private int[] GetIndicesFromFlatIndex(int flatIndex)
        {
            // Массив для хранения индексов для каждого измерения
            int[] indices = new int[Rank];

            // Преобразуем плоский индекс в многомерный, начиная с первой оси
            for (int i = 0; i < Rank; i++)
            {
                // Определяем индекс для текущего измерения
                indices[i] = flatIndex % Shape[i];

                // Снижаем плоский индекс для следующего измерения
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
