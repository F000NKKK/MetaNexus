using MetaNexus.Lib.NeuralNetwork.Tensor.Abstractions;

namespace MetaNexus.Lib.NeuralNetwork.Tensor
{
    public partial struct Tensor : ITensorNormalizationOperations
    {
        public Tensor BatchNormalize(Tensor mean, Tensor variance)
        {
            throw new NotImplementedException();
        }

        public Tensor Normalize()
        {
            throw new NotImplementedException();
        }
    }
}
