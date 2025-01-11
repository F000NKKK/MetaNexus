using MetaNexus.Lib.NeuralNetwork.Math.Normalizers.Abstractions;

namespace MetaNexus.Lib.NeuralNetwork.Math.Normalizers
{
    public class MinMaxNormalizer : Normalizer
    {
        private float _min;
        private float _max;

        public MinMaxNormalizer(float[] data)
        {
            _min = data.Min();
            _max = data.Max();
        }

        public override float[] Normalize(float[] data)
        {
            return data.Select(x => (x - _min) / (_max - _min)).ToArray();
        }

        public override float[] Denormalize(float[] normalizedData)
        {
            return normalizedData.Select(x => x * (_max - _min) + _min).ToArray();
        }
    }
}
