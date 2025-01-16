using MetaNexus.Lib.NeuralNetwork.Tensors.Abstractions;
using System;
using System.Linq;

namespace MetaNexus.Lib.NeuralNetwork.Tensors
{
    public partial struct Tensor : ITensorBroadcastingOperations
    {
        // Проверка возможности broadcasting двух тензоров
        public bool CanBroadcast(Tensor other)
        {
            int maxLength = Math.Max(this.Shape.Length, other.Shape.Length);

            // Перебор осей тензоров с добавлением осей размера 1 при необходимости
            for (int i = 0; i < maxLength; i++)
            {
                int dim1 = i < this.Shape.Length ? this.Shape[i] : 1;
                int dim2 = i < other.Shape.Length ? other.Shape[i] : 1;

                if (dim1 != dim2 && dim1 != 1 && dim2 != 1)
                {
                    return false;
                }
            }
            return true;
        }

        public Tensor BroadcastAdd(Tensor other)
        {
            if (!CanBroadcast(other))
            {
                throw new InvalidOperationException("Невозможно выполнить операцию с трансляцией для этих тензоров.");
            }

            // Преобразуем оба тензора в форму, совместимую для broadcasting
            Tensor expandedTensor1 = this.ExpandToBroadcast(other.Shape);
            Tensor expandedTensor2 = other.ExpandToBroadcast(this.Shape);

            // Выводим информацию о формах
            Console.WriteLine($"Expanded Tensor 1 Shape: {string.Join(",", expandedTensor1.Shape)}");
            Console.WriteLine($"Expanded Tensor 2 Shape: {string.Join(",", expandedTensor2.Shape)}");

            // Проверка на совпадение форм после расширения
            if (!expandedTensor1.Shape.SequenceEqual(expandedTensor2.Shape))
            {
                throw new InvalidOperationException("Формы расширенных тензоров не совпадают после трансляции.");
            }

            return expandedTensor1.Add(expandedTensor2);
        }

        // Операция вычитания с трансляцией
        public Tensor BroadcastSubtract(Tensor other)
        {
            if (!CanBroadcast(other))
            {
                throw new InvalidOperationException("Невозможно выполнить операцию с трансляцией для этих тензоров.");
            }

            Tensor expandedTensor1 = this.ExpandToBroadcast(other.Shape);
            Tensor expandedTensor2 = other.ExpandToBroadcast(this.Shape);

            return expandedTensor1.Subtract(expandedTensor2);
        }

        // Операция умножения с трансляцией
        public Tensor BroadcastMultiply(Tensor other)
        {
            if (!CanBroadcast(other))
            {
                throw new InvalidOperationException("Невозможно выполнить операцию с трансляцией для этих тензоров.");
            }

            Tensor expandedTensor1 = this.ExpandToBroadcast(other.Shape);
            Tensor expandedTensor2 = other.ExpandToBroadcast(this.Shape);

            return expandedTensor1.Multiply(expandedTensor2);
        }

        // Операция деления с трансляцией
        public Tensor BroadcastDivide(Tensor other)
        {
            if (!CanBroadcast(other))
            {
                throw new InvalidOperationException("Невозможно выполнить операцию с трансляцией для этих тензоров.");
            }

            Tensor expandedTensor1 = this.ExpandToBroadcast(other.Shape);
            Tensor expandedTensor2 = other.ExpandToBroadcast(this.Shape);

            // Проверка деления на ноль
            if (expandedTensor2.Data.Contains(0))
            {
                throw new DivideByZeroException("Деление на ноль невозможно.");
            }

            return expandedTensor1.Divide(expandedTensor2);
        }

        // Расширение текущего тензора до требуемой формы для broadcasting
        public Tensor ExpandToBroadcast(int[] targetShape)
        {
            int[] sourceShape = this.Shape;
            int[] expandedShape = new int[targetShape.Length];

            // Расширяем форму с учетом targetShape
            for (int i = 0; i < targetShape.Length; i++)
            {
                // Если размер оси в targetShape больше чем в sourceShape, или в sourceShape = 1
                if (i < sourceShape.Length)
                {
                    expandedShape[i] = Math.Max(sourceShape[i], targetShape[i]);
                }
                else
                {
                    expandedShape[i] = targetShape[i];  // Если эта ось существует только в targetShape
                }
            }

            // Пересчитываем количество элементов
            int totalElements = expandedShape.Aggregate(1, (a, b) => a * b);
            float[] expandedData = new float[totalElements];

            // Страйды для исходных и расширенных форм
            int[] sourceStrides = ComputeStrides(sourceShape);
            int[] expandedStrides = ComputeStrides(expandedShape);

            // Расширяем данные с учетом новых размеров
            for (int i = 0; i < totalElements; i++)
            {
                int expandedIndex = i;
                int sourceIndex = 0;

                // Для каждой оси вычисляем, как индекс из расширенного тензора маппится в исходные данные
                for (int dim = expandedShape.Length - 1; dim >= 0; dim--)
                {
                    int expandedDimIndex = (expandedIndex / expandedStrides[dim]) % expandedShape[dim];
                    int sourceDimIndex = dim < sourceShape.Length && sourceShape[dim] != 1
                        ? expandedDimIndex % sourceShape[dim]
                        : 0;

                    sourceIndex += sourceDimIndex * (dim < sourceStrides.Length ? sourceStrides[dim] : 1);
                }

                expandedData[i] = this._data.Span[sourceIndex];
            }

            return new Tensor(expandedShape, expandedData);
        }


        // Вычисление шагов (strides) для текущей формы
        private int[] ComputeStrides(int[] shape)
        {
            int[] strides = new int[shape.Length];
            int stride = 1;

            for (int i = shape.Length - 1; i >= 0; i--)
            {
                strides[i] = stride;
                stride *= shape[i];
            }

            return strides;
        }
    }
}
