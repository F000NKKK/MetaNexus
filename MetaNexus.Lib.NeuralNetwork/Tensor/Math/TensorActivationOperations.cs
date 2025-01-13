using MetaNexus.Lib.NeuralNetwork.Tensor.Math.Abstractions;
using System.Numerics;

namespace MetaNexus.Lib.NeuralNetwork.Math.Tensor
{
    public partial struct Tensor<T> : ITensorActivationOperations<T> where T : INumber<T>
    {
        public Tensor<T> ApplyReLU()
        {
            throw new NotImplementedException();
        }

        public Tensor<T> ApplySigmoid()
        {
            throw new NotImplementedException();
        }

        public Tensor<T> ApplyTanh()
        {
            throw new NotImplementedException();
        }
    }
}
