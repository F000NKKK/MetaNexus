using MetaNexus.Lib.NeuralNetwork.Tensor.Abstractions;

namespace MetaNexus.Lib.NeuralNetwork.Tensor
{
    public partial struct Tensor : ITensorShapeOperations
    {
        public Tensor Reshape(int[] newShape)
        {
            throw new NotImplementedException();
        }

        public Tensor Transpose()
        {
            throw new NotImplementedException();
        }
    }
}
