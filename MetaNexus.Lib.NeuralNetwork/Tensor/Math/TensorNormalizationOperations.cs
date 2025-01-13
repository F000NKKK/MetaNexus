using MetaNexus.Lib.NeuralNetwork.Tensor.Math.Abstractions;
using System.Numerics;

namespace MetaNexus.Lib.NeuralNetwork.Math.Tensor
{
    public partial struct Tensor<T> : ITensorNormalizationOperations<T> where T : INumber<T>
    {
        public Tensor<T> BatchNormalize(Tensor<T> mean, Tensor<T> variance)
        {
            throw new NotImplementedException();
        }

        public Tensor<T> Normalize()
        {
            throw new NotImplementedException();
        }
    }
}
