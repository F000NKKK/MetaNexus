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
        }

        /// <summary>
        /// Прямой проход через слой пакетной нормализации.
        /// </summary>
        /// <param name="input">Входной тензор.</param>
        /// <returns>Нормализованный выходной тензор.</returns>
        public override Tensor Forward(Tensor input)
        {
            throw new NotImplementedException();
        }
    }
}
