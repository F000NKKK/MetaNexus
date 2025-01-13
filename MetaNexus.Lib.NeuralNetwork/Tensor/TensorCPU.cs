using MetaNexus.Lib.NeuralNetwork.Tensor.Abstractions;
using System.Numerics;

namespace MetaNexus.Lib.NeuralNetwork.Tensor
{
    internal struct TensorCPU<T> : ITensorDevice<T> where T : INumber<T>
    {
        public TensorCPU(int[] shape)
        {
        }
        public void ExecuteKernel(string kernelCode)
        {
            throw new NotImplementedException();
        }

        public T[] GetDataFromDevice()
        {
            throw new NotImplementedException();
        }

        public void TransferToDevice(T[] data)
        {
            throw new NotImplementedException();
        }
    }
}
