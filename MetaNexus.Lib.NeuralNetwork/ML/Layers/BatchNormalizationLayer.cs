using MetaNexus.Lib.NeuralNetwork.ML.Layers.Abstractions;

namespace MetaNexus.Lib.NeuralNetwork.ML.Layers
{
    public class BatchNormalizationLayer : Layer
    {
        private int _size;
        private float[] _gamma;
        private float[] _beta;
        private float[] _mean;
        private float[] _variance;

        public BatchNormalizationLayer(int size) : base(size)
        {
            _size = size;
            _gamma = new float[size]; // Гамма (масштабирование)
            _beta = new float[size];  // Бета (сдвиг)
            _mean = new float[size];  // Среднее
            _variance = new float[size]; // Дисперсия

            // Инициализация гамма и бета
            for (int i = 0; i < size; i++)
            {
                _gamma[i] = 1.0f;
                _beta[i] = 0.0f;
            }
        }

        public override float[] Forward(float[] input)
        {
            // Нормализуем данные: (x - mean) / sqrt(variance + epsilon)
            float epsilon = 1e-8f;
            float[] normalized = new float[input.Length];

            for (int i = 0; i < _size; i++)
            {
                // Пример нормализации: тут предполагается, что mean и variance заданы заранее
                normalized[i] = (_gamma[i] * (input[i] - _mean[i])) / MathF.Sqrt(_variance[i] + epsilon) + _beta[i];
            }

            return normalized;
        }
    }
}
