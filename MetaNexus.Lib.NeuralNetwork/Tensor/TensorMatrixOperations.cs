using MetaNexus.Lib.NeuralNetwork.Tensor.Abstractions;
using System.Numerics;

namespace MetaNexus.Lib.NeuralNetwork.Tensor
{
    public partial struct Tensor<T> : ITensorMatrixOperations<T> where T : INumber<T>
    {
        public Tensor<T> MatrixMultiply(Tensor<T> other)
        {
            throw new NotImplementedException();
        }
    }
}
