using MetaNexus.Lib.NeuralNetwork.Tensors.Abstractions;

namespace MetaNexus.Lib.NeuralNetwork.Tensors
{
    public partial struct Tensor : ITensorNormalizationOperations
    {
        public Tensor Normalize()
        {
            float mean = 0;
            float variance = 0;

            foreach (var value in _data.Span.ToArray())
            {
                mean += value;
            }
            mean /= Size;

            foreach (var value in _data.Span.ToArray())
            {
                variance += (value - mean) * (value - mean);
            }
            variance /= Size;
            float stdDev = (float)Math.Sqrt(variance);

            var result = new Tensor(_shape);
            for (int i = 0; i < Size; i++)
            {
                result._data.Span.ToArray()[i] = (this._data.Span.ToArray()[i] - mean) / stdDev;
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
                result._data.Span.ToArray()[i] = (this._data.Span.ToArray()[i] - mean._data.Span.ToArray()[i]) / (float)Math.Sqrt(variance._data.Span.ToArray()[i] + 1e-8f);
            }
            return result;
        }

        public Tensor MinMaxNormalize()
        {
            float min = _data.Span.ToArray()[0];
            float max = _data.Span.ToArray()[0];

            foreach (var value in _data.Span.ToArray())
            {
                if (value < min) min = value;
                if (value > max) max = value;
            }

            var result = new Tensor(_shape);
            for (int i = 0; i < Size; i++)
            {
                result._data.Span.ToArray()[i] = (_data.Span.ToArray()[i] - min) / (max - min);
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
                result._data.Span.ToArray()[i] = (this._data.Span.ToArray()[i] - mean._data.Span.ToArray()[i]) / (float)Math.Sqrt(variance._data.Span.ToArray()[i] + 1e-8f);
            }
            return result;
        }

        public Tensor Standardize()
        {
            float mean = 0;
            float variance = 0;

            foreach (var value in _data.Span.ToArray())
            {
                mean += value;
            }
            mean /= Size;

            foreach (var value in _data.Span.ToArray())
            {
                variance += (value - mean) * (value - mean);
            }
            variance /= Size;
            float stdDev = (float)Math.Sqrt(variance);

            var result = new Tensor(_shape);
            for (int i = 0; i < Size; i++)
            {
                result._data.Span.ToArray()[i] = (this._data.Span.ToArray()[i] - mean) / stdDev;
            }

            return result;
        }

        public Tensor LabelNormalize(int numClasses)
        {
            var result = new Tensor(_shape);
            for (int i = 0; i < Size; i++)
            {
                result._data.Span.ToArray()[i] = this._data.Span.ToArray()[i] / numClasses;
            }

            return result;
        }
    }
}
