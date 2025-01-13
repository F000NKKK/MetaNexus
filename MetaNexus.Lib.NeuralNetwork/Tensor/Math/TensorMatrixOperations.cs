using MetaNexus.Lib.NeuralNetwork.Tensor.Math.Abstractions;
using System.Numerics;

namespace MetaNexus.Lib.NeuralNetwork.Math.Tensor
{
    public partial struct Tensor<T> : ITensorMatrixOperations<T> where T : INumber<T>
    {
        public Tensor<T> MatrixMultiply(Tensor<T> other)
        {
            throw new NotImplementedException();
        }
    }
}
