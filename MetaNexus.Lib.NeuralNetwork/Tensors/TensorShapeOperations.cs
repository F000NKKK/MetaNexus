using MetaNexus.Lib.NeuralNetwork.Tensors.Abstractions;

namespace MetaNexus.Lib.NeuralNetwork.Tensors
{
    public partial struct Tensor : ITensorShapeOperations
    {
        public Tensor Transpose()
        {
            if (_shape.Count() != 2)
                throw new InvalidOperationException("Транспонирование поддерживается только для двумерных тензоров.");

            var newShape = new int[] { _shape[1], _shape[0] };
            var result = new Tensor(newShape);

            for (int i = 0; i < _shape[0]; i++)
            {
                for (int j = 0; j < _shape[1]; j++)
                {
                    result[j, i] = this[i, j];
                }
            }

            return result;
        }

        public Tensor Clip(float minValue, float maxValue) => new Tensor(_shape, this.Data.Select(x => Math.Clamp(x, minValue, maxValue)).ToArray());


        public Tensor Reshape(int[] newShape)
        {
            int newSize = 1;
            foreach (var dim in newShape)
            {
                newSize *= dim;
            }

            if (newSize != Size)
                throw new InvalidOperationException("Новая форма должна содержать такое же количество элементов, как и старая.");

            var result = new Tensor(newShape);
            Array.Copy(_data.Span.ToArray(), result._data.Span.ToArray(), Size);
            return result;
        }

        public Tensor[] Split(int axis, int parts)
        {
            if (axis < 0 || axis >= _shape.Length)
                throw new ArgumentException("Ось должна быть в пределах размерности тензора.");

            if (_shape[axis] % parts != 0)
                throw new ArgumentException("Размерность вдоль оси не делится на количество частей.");

            int splitSize = _shape[axis] / parts;
            var result = new Tensor[parts];

            for (int i = 0; i < parts; i++)
            {
                int[] newShape = (int[])_shape.Clone();
                newShape[axis] = splitSize;

                var splitTensor = new Tensor(newShape);
                for (int j = 0; j < splitSize; j++)
                {
                    for (int k = 0; k < Size / _shape[axis]; k++)
                    {
                        splitTensor[j, k] = this[i * splitSize + j, k];
                    }
                }
                result[i] = splitTensor;
            }

            return result;
        }

        public Tensor TransposeAxes(int[] newOrder)
        {
            if (newOrder.Length != _shape.Length)
                throw new InvalidOperationException("Новый порядок осей должен быть той же длины, что и текущий.");

            bool[] visited = new bool[_shape.Length];
            foreach (var index in newOrder)
            {
                if (index < 0 || index >= _shape.Length || visited[index])
                    throw new InvalidOperationException("Неверный порядок осей (переопределение или неправильный индекс).");

                visited[index] = true;
            }

            var resultShape = new int[_shape.Length];
            for (int i = 0; i < _shape.Length; i++)
            {
                resultShape[i] = _shape[newOrder[i]];
            }

            var result = new Tensor(resultShape);
            int[] indices = new int[_shape.Length];
            int flatIndex;

            Tensor tensor = this;

            for (int i = 0; i < Size; i++)
            {
                indices = GetIndicesFromFlatIndex(i);
                flatIndex = GetFlatIndexFromNewOrder(indices, newOrder);
                result._data.Span.ToArray()[flatIndex] = tensor._data.Span.ToArray()[i];
            }

            return result;
        }
    }
}
