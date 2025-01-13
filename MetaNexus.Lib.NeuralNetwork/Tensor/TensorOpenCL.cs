using MetaNexus.Lib.NeuralNetwork.Tensor.Abstractions;
using System.Numerics;

namespace MetaNexus.Lib.NeuralNetwork.Tensor
{
    internal struct TensorOpenCL<T> : ITensorGPU<T> where T : INumber<T>
    {
        public TensorOpenCL()
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