using MetaNexus.Lib.NeuralNetwork.Tensors;
using MetaNexus.Lib.NeuralNetwork.ML.Layers.Abstractions;
using System;

namespace MetaNexus.Lib.NeuralNetwork.ML.Layers
{
    /// <summary>
    /// Класс для слоя пакетной нормализации.
    /// </summary>
    public class BatchNormalizationLayer : Layer
    {
        private int _size;
        private Tensor _gamma;
        private Tensor _beta;
        private Tensor _mean;
        private Tensor _variance;

        /// <summary>
        /// Конструктор для слоя пакетной нормализации.
        /// </summary>
        /// <param name="size">Размер слоя (количество нейронов).</param>
        public BatchNormalizationLayer(int size) : base(size)
        {
            _size = size;
            _gamma = new Tensor(new int[] { size }); // Гамма (масштабирование)
            _beta = new Tensor(new int[] { size });  // Бета (сдвиг)
            _mean = new Tensor(new int[] { size });  // Среднее
            _variance = new Tensor(new int[] { size }); // Дисперсия

            // Инициализация гамма и бета
            for (int i = 0; i < size; i++)
            {
                _gamma[0, i] = 1.0f;
                _beta[0, i] = 0.0f;
            }
        }

        /// <summary>
        /// Прямой проход через слой пакетной нормализации.
        /// </summary>
        /// <param name="input">Входной тензор.</param>
        /// <returns>Нормализованный выходной тензор.</returns>
        public override Tensor Forward(Tensor input)
        {
            // Используем методы нормализации из Tensor
            float epsilon = 1e-8f;

            // Применение пакетной нормализации (x - mean) / sqrt(variance + epsilon)
            // Мы предполагаем, что _mean и _variance уже вычислены (или их можно вычислить в процессе)
            Tensor normalized = input.BatchNormalize(_mean, _variance);

            // Масштабирование и сдвиг с использованием гаммы и беты
            normalized = _gamma * normalized + _beta;

            return normalized;
        }
    }
}
