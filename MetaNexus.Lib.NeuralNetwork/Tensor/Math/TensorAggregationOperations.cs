﻿using MetaNexus.Lib.NeuralNetwork.Tensor.Math.Abstractions;
using System.Numerics;

namespace MetaNexus.Lib.NeuralNetwork.Math.Tensor
{
    public partial struct Tensor<T> : ITensorAggregationOperations<T> where T : INumber<T>
    {
        public T Max()
        {
            throw new NotImplementedException();
        }

        public T Mean()
        {
            throw new NotImplementedException();
        }

        public T Min()
        {
            throw new NotImplementedException();
        }

        public T Sum()
        {
            throw new NotImplementedException();
        }
    }
}
