using MetaNexus.Lib.NeuralNetwork.Tensor.Abstractions;
using System.Numerics;

namespace MetaNexus.Lib.NeuralNetwork.Tensor
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
