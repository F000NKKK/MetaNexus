using MetaNexus.Lib.NeuralNetwork.Tensor.Abstractions;
using System.Numerics;

namespace MetaNexus.Lib.NeuralNetwork.Tensor
{
    internal struct TensorCUDA<T> : ITensorGPU<T> where T : INumber<T>
    {
        public TensorCUDA()
        {
        }

        public void ExecuteKernel(string kernelCode)
        {
            throw new NotImplementedException();
        }

        public T[] GetDataFromGPU()
        {
            throw new NotImplementedException();
        }

        public void TransferToGPU(T[] data)
        {
            throw new NotImplementedException();
        }
    }
}