using MetaNexus.Lib.NeuralNetwork.Tensors.Abstractions;

namespace MetaNexus.Lib.NeuralNetwork.Tensors
{
    public partial struct Tensor : ITensorNormalizationOperations
    {
        public Tensor Normalize()
        {
            float mean = 0;
            float variance = 0;

            foreach (var value in _data.Span)
            {
                mean += value;
            }
            mean /= Size;

            foreach (var value in _data.Span)
            {
                variance += (value - mean) * (value - mean);
            }
            variance /= Size;
            float stdDev = MathF.Sqrt(variance);

            var result = new Tensor(_shape);
            for (int i = 0; i < Size; i++)
            {
                result._data.Span[i] = (this._data.Span[i] - mean) / stdDev;
            }

            return result;
        }

        public Tensor BatchNormalize(Tensor mean, Tensor variance)
        {
            if (mean.Size != variance.Size || mean.Size != this.Size)
                throw new ArgumentException("Размеры тензора, среднего и дисперсии должны совпадать.");

            var result = new Tensor(_shape);
            for (int i = 0; i < Size; i++)
            {
                result._data.Span[i] = (this._data.Span[i] - mean._data.Span[i]) / MathF.Sqrt(variance._data.Span[i] + 1e-8f);
            }
            return result;
        }

        public Tensor MinMaxNormalize()
        {
            float min = _data.Span[0];
            float max = _data.Span[0];

            foreach (var value in _data.Span)
            {
                if (value < min) min = value;
                if (value > max) max = value;
            }

            var result = new Tensor(_shape);
            for (int i = 0; i < Size; i++)
            {
                result._data.Span[i] = (_data.Span[i] - min) / (max - min);
            }

            return result;
        }

        public Tensor ChannelNormalize(Tensor mean, Tensor variance)
        {
            if (mean.Size != variance.Size || mean.Size != this.Size)
                throw new ArgumentException("Размеры тензора, среднего и дисперсии должны совпадать.");

            var result = new Tensor(_shape);
            for (int i = 0; i < Size; i++)
            {
                result._data.Span[i] = (this._data.Span[i] - mean._data.Span[i]) / MathF.Sqrt(variance._data.Span[i] + 1e-8f);
            }
            return result;
        }

        public Tensor Standardize()
        {
            float mean = 0;
            float variance = 0;

            // Вычисляем среднее значение
            foreach (var value in _data.Span)
            {
                mean += value;
            }
            mean /= Size;

            // Вычисляем дисперсию
            foreach (var value in _data.Span)
            {
                variance += (value - mean) * (value - mean);
            }
            variance /= Size;

            // Вычисляем стандартное отклонение
            float stdDev = MathF.Sqrt(variance);

            // Если стандартное отклонение равно 0 (все элементы одинаковы), возвращаем исходный тензор
            if (stdDev == 0)
            {
                return this;
            }

            // Создаем новый тензор для результата
            var result = new Tensor(_shape);

            // Стандартизируем значения
            for (int i = 0; i < Size; i++)
            {
                result._data.Span[i] = (this._data.Span[i] - mean) / stdDev;
            }

            return result;
        }


        public Tensor LabelNormalize(int numClasses)
        {
            var result = new Tensor(_shape);
            for (int i = 0; i < Size; i++)
            {
                result._data.Span[i] = this._data.Span[i] / numClasses;
            }

            return result;
        }
    }
}
