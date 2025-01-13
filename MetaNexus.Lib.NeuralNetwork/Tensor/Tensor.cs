using MetaNexus.Lib.NeuralNetwork.Tensor.Abstractions;
using MetaNexus.Lib.NeuralNetwork.Tensor.Math.Abstractions;
using Newtonsoft.Json;
using System;
using System.Numerics;

namespace MetaNexus.Lib.NeuralNetwork.Math.Tensor
{
    /// <summary>
    /// Структура Tensor представляет многомерный массив числовых данных с поддержкой операций над тензорами.
    /// </summary>
    public partial struct Tensor<T> : ITensor<T> where T : INumber<T>
    {
        public T this[params int[] indices] { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }

        public int[] Shape => throw new NotImplementedException();

        public int Rank => throw new NotImplementedException();

        public int Size => throw new NotImplementedException();

        public Tensor<T> Apply(Func<T, T> func)
        {
            throw new NotImplementedException();
        }

        public Tensor<T> Clone()
        {
            throw new NotImplementedException();
        }

        public Tensor<TTarget> ConvertTo<TTarget>() where TTarget : INumber<TTarget>
        {
            throw new NotImplementedException();
        }

        public T[] Flatten()
        {
            throw new NotImplementedException();
        }

        public bool IsEmpty()
        {
            throw new NotImplementedException();
        }
    }
}
