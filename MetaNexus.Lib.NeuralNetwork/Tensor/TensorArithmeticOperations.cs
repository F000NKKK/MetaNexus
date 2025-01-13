using MetaNexus.Lib.NeuralNetwork.Tensor.Abstractions;
using System.Numerics;

namespace MetaNexus.Lib.NeuralNetwork.Tensor
{
    public partial struct Tensor<T> : ITensorArithmeticOperations<T> where T : INumber<T>
    {
        public Tensor<T> Add(Tensor<T> other)
        {
            throw new NotImplementedException();
        }

        public Tensor<T> Add(T scalar)
        {
            throw new NotImplementedException();
        }

        public Tensor<T> Divide(Tensor<T> other)
        {
            throw new NotImplementedException();
        }

        public Tensor<T> Divide(T scalar)
        {
            throw new NotImplementedException();
        }

        public Tensor<T> Multiply(Tensor<T> other)
        {
            throw new NotImplementedException();
        }

        public Tensor<T> Multiply(T scalar)
        {
            throw new NotImplementedException();
        }

        public Tensor<T> Subtract(Tensor<T> other)
        {
            throw new NotImplementedException();
        }

        public Tensor<T> Subtract(T scalar)
        {
            throw new NotImplementedException();
        }
    }
}
