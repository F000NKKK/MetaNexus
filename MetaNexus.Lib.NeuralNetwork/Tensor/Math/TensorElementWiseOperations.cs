using MetaNexus.Lib.NeuralNetwork.Tensor.Math.Abstractions;
using System.Numerics;

namespace MetaNexus.Lib.NeuralNetwork.Math.Tensor
{
    public partial struct Tensor<T> : ITensorElementWiseOperations<T> where T : INumber<T>
    {
        public Tensor<T> ElementWiseAdd(Tensor<T> other)
        {
            throw new NotImplementedException();
        }

        public Tensor<T> ElementWiseDivide(Tensor<T> other)
        {
            throw new NotImplementedException();
        }

        public Tensor<T> ElementWiseMultiply(Tensor<T> other)
        {
            throw new NotImplementedException();
        }

        public Tensor<T> ElementWiseSubtract(Tensor<T> other)
        {
            throw new NotImplementedException();
        }
    }
}
