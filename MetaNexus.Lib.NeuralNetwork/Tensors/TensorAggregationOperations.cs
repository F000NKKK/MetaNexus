﻿using MetaNexus.Lib.NeuralNetwork.Tensors.Abstractions;

namespace MetaNexus.Lib.NeuralNetwork.Tensors
{
    public partial struct Tensor : ITensorAggregationOperations
    {
        public float Max()
        {
            if (_data == null || _data.Length == 0)
                throw new InvalidOperationException("Тензор пуст");

            return _data.Max();
        }

        public float Min()
        {
            if (_data == null || _data.Length == 0)
                throw new InvalidOperationException("Тензор пуст");

            return _data.Min();
        }

        public float Avg()
        {
            if (_data == null || _data.Length == 0)
                throw new InvalidOperationException("Тензор пуст");

            return _data.Average();
        }

        public float Sum()
        {
            if (_data == null || _data.Length == 0)
                throw new InvalidOperationException("Тензор пуст");

            return _data.Sum();
        }
    }
}