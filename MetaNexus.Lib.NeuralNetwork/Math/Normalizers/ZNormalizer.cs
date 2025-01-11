using MetaNexus.Lib.NeuralNetwork.Math.Normalizers.Abstractions;

namespace MetaNexus.Lib.NeuralNetwork.Math.Normalizers
{
    public class ZNormalizer : Normalizer
    {
        private float _mean;
        private float _stdDev;

        public ZNormalizer(float[] data)
        {
            _mean = data.Average();
            _stdDev = MathF.Sqrt(data.Select(x => (x - _mean) * (x - _mean)).Average());
        }

        public override float[] Normalize(float[] data)
        {
            return data.Select(x => (x - _mean) / _stdDev).ToArray();
        }

        public override float[] Denormalize(float[] normalizedData)
        {
            return normalizedData.Select(x => x * _stdDev + _mean).ToArray();
        }
    }
}
