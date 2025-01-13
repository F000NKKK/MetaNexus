using MetaNexus.Lib.NeuralNetwork.Tensor.Abstractions;
using System;

namespace MetaNexus.Lib.NeuralNetwork.Tensor
{
    public partial struct Tensor : ITensorNormalizationOperations
    {
        public Tensor Normalize()
        {
            float mean = 0;
            float variance = 0;

            foreach (var value in _data)
            {
                mean += value;
            }
            mean /= Size;

            foreach (var value in _data)
            {
                variance += (value - mean) * (value - mean);
            }
            variance /= Size;
            float stdDev = (float)Math.Sqrt(variance);

            var result = new Tensor(_shape);
            for (int i = 0; i < Size; i++)
            {
                result._data[i] = (this._data[i] - mean) / stdDev;
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
                result._data[i] = (this._data[i] - mean._data[i]) / (float)Math.Sqrt(variance._data[i] + 1e-8f);
            }
            return result;
        }

        public Tensor MinMaxNormalize()
        {
            float min = _data[0];
            float max = _data[0];

            foreach (var value in _data)
            {
                if (value < min) min = value;
                if (value > max) max = value;
            }

            var result = new Tensor(_shape);
            for (int i = 0; i < Size; i++)
            {
                result._data[i] = (_data[i] - min) / (max - min);
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
                result._data[i] = (this._data[i] - mean._data[i]) / (float)Math.Sqrt(variance._data[i] + 1e-8f);
            }
            return result;
        }

        public Tensor Standardize()
        {
            float mean = 0;
            float variance = 0;

            foreach (var value in _data)
            {
                mean += value;
            }
            mean /= Size;

            foreach (var value in _data)
            {
                variance += (value - mean) * (value - mean);
            }
            variance /= Size;
            float stdDev = (float)Math.Sqrt(variance);

            var result = new Tensor(_shape);
            for (int i = 0; i < Size; i++)
            {
                result._data[i] = (this._data[i] - mean) / stdDev;
            }

            return result;
        }

        public Tensor LabelNormalize(int numClasses)
        {
            var result = new Tensor(_shape);
            for (int i = 0; i < Size; i++)
            {
                result._data[i] = this._data[i] / numClasses;
            }

            return result;
        }
    }
}
