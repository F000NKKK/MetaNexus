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
        private Tensor _gamma;
        private Tensor _beta;
        private Tensor _avg;
        private Tensor _variance;
        private bool _training;

        public BatchNormalizationLayer(int inputSize, int size, bool training = true) : base(inputSize, size)
        {
            _gamma = new Tensor(new int[] { size }); // Гамма (масштабирование)
            _beta = new Tensor(new int[] { size });  // Бета (сдвиг)
            _avg = new Tensor(new int[] { size });  // Среднее
            _variance = new Tensor(new int[] { size }); // Дисперсия
            _training = training;  // Флаг для определения, находимся ли в процессе обучения

            // Инициализация гамма и бета
            for (int i = 0; i < size; i++)
            {
                _gamma[i] = 1.0f;
                _beta[i] = 0.0f;
            }

            // Инициализация среднего и дисперсии
            for (int i = 0; i < size; i++)
            {
                _avg[i] = 0.0f;
                _variance[i] = 1.0f; // Инициализируем дисперсию как 1, так как для нормализации это безопасно
            }
        }

        /// <summary>
        /// Прямой проход через слой пакетной нормализации.
        /// </summary>
        /// <param name="input">Входной тензор.</param>
        /// <returns>Нормализованный выходной тензор.</returns>
        public override Tensor Forward(Tensor input)
        {
            Tensor output = new Tensor(input.Shape);

            if (_training)
            {
                // В процессе обучения нормализуем данные по мини-батчу
                Tensor mean = input.Avg(0); // Среднее по 0-ой оси (по каждому каналу)
                Tensor variance = input.Variance(0); // Дисперсия по 0-ой оси

                // Нормализуем вход по мини-батчу
                output = input.BatchNormalize(mean, variance);

                // Обновляем среднее и дисперсию для использования при инференсе
                _avg = _avg * 0.9f + mean * 0.1f;  // Используем скользящее среднее
                _variance = _variance * 0.9f + variance * 0.1f;  // Используем скользящее среднее
            }
            else
            {
                // Для инференса используем глобальное среднее и дисперсию
                output = input.BatchNormalize(_avg, _variance);
            }

            // Применяем гамма и бета
            output = output * _gamma + _beta;

            return output;
        }
    }
}
