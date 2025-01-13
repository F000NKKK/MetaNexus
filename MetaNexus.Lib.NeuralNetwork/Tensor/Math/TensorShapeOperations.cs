using MetaNexus.Lib.NeuralNetwork.Tensor.Math.Abstractions;
using System.Numerics;

namespace MetaNexus.Lib.NeuralNetwork.Math.Tensor
{
    public partial struct Tensor<T> : ITensorShapeOperations<T> where T : INumber<T>
    {
        public Tensor<T> Reshape(int[] newShape)
        {
            throw new NotImplementedException();
        }

        public Tensor<T> Transpose()
        {
            throw new NotImplementedException();
        }
    }
}
