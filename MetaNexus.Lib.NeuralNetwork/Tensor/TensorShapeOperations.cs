using MetaNexus.Lib.NeuralNetwork.Tensor.Abstractions;
using System;

namespace MetaNexus.Lib.NeuralNetwork.Tensor
{
    public partial struct Tensor : ITensorShapeOperations
    {
        // Транспонирование тензора (для матриц)
        public Tensor Transpose()
        {
            if (_shape.Length != 2)
                throw new InvalidOperationException("Транспонирование поддерживается только для двумерных тензоров.");

            // Меняем местами оси 0 и 1 для матрицы
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

        // Изменение формы тензора
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
            Array.Copy(_data, result._data, Size);
            return result;
        }

        // Разбиение тензора на несколько частей вдоль указанной оси
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

        // Изменение порядка осей тензора
        public Tensor TransposeAxes(int[] newOrder)
        {
            if (newOrder.Length != _shape.Length)
                throw new InvalidOperationException("Новый порядок осей должен быть той же длины, что и текущий.");

            // Проверяем, что новый порядок является перестановкой
            bool[] visited = new bool[_shape.Length];
            foreach (var index in newOrder)
            {
                if (index < 0 || index >= _shape.Length || visited[index])
                    throw new InvalidOperationException("Неверный порядок осей (переопределение или неправильный индекс).");

                visited[index] = true;
            }

            // Генерация нового тензора с изменённым порядком осей
            var resultShape = new int[_shape.Length];
            for (int i = 0; i < _shape.Length; i++)
            {
                resultShape[i] = _shape[newOrder[i]];
            }

            var result = new Tensor(resultShape);
            int[] indices = new int[_shape.Length];
            int flatIndex;

            // Создаем копию тензора, чтобы избежать ошибки доступа к 'this' внутри лямбда-выражений
            Tensor tensor = this;

            for (int i = 0; i < Size; i++)
            {
                GetIndicesFromFlatIndex(i, indices);
                flatIndex = GetFlatIndexFromNewOrder(indices, newOrder);
                result._data[flatIndex] = tensor._data[i];
            }

            return result;
        }

        // Получение индексов для плоского массива
        private void GetIndicesFromFlatIndex(int flatIndex, int[] indices)
        {
            int remainingIndex = flatIndex;
            for (int i = _shape.Length - 1; i >= 0; i--)
            {
                indices[i] = remainingIndex % _shape[i];
                remainingIndex /= _shape[i];
            }
        }

        // Получение нового плоского индекса с учётом нового порядка осей
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
    }
}
