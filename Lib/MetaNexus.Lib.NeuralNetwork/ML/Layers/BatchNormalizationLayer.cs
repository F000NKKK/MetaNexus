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
        private bool _training;
        private Tensor γ; // Параметр масштабирования
        private Tensor β; // Параметр смещения
        private Tensor movingMean; // Скользящее среднее для оценки среднего
        private Tensor movingVariance; // Скользящее среднее для оценки дисперсии
        private float momentum; // Коэффициент импульса для обновления скользящих значений
        private float epsilon; // Маленькое значение для числовой стабильности

        public BatchNormalizationLayer(int inputSize, int size, bool training = true) : base(inputSize, size)
        {
            _training = training;
            epsilon = 1e-5f;
            momentum = 0.9f;

            // Инициализация параметров γ и β
            γ = new Tensor(new[] { size }).Fill(1.0f); // Масштабирование, начальное значение = 1
            β = new Tensor(new[] { size }).Fill(0.0f); // Смещение, начальное значение = 0

            // Инициализация скользящих значений
            movingMean = new Tensor(new[] { 1, size }).Fill(0.0f); // Среднее
            movingVariance = new Tensor(new[] { 1, size }).Fill(1.0f); // Дисперсия
        }

        /// <summary>
        /// Прямой проход через слой пакетной нормализации.
        /// </summary>
        /// <param name="input">Входной тензор.</param>
        /// <returns>Нормализованный выходной тензор.</returns>
        public override Tensor Forward(Tensor input)
        {
            if (_training)
            {
                Console.WriteLine("Выполняется прямой проход в режиме обучения.");

                // Вычисляем среднее и дисперсию по входным данным
                Tensor mean = input.Avg(axis: 0);
                Console.WriteLine($"Среднее по batch: {mean}");

                Tensor variance = input.Variance(axis: 0);
                Console.WriteLine($"Дисперсия по batch: {variance}");

                // Нормализация входных данных
                Tensor normalized = (input - mean) / (variance + epsilon).Sqrt();
                Console.WriteLine($"Нормализованные данные: {normalized}");

                // Обновляем скользящие значения
                Tensor updatedMovingMean = movingMean * momentum + mean * (1 - momentum);
                Console.WriteLine($"Обновленное скользящее среднее: {updatedMovingMean}");
                movingMean = updatedMovingMean;

                Tensor updatedMovingVariance = movingVariance * momentum + variance * (1 - momentum);
                Console.WriteLine($"Обновленная скользящая дисперсия: {updatedMovingVariance}");
                movingVariance = updatedMovingVariance;

                // Масштабируем и смещаем нормализованные данные
                Tensor output = γ * normalized + β;
                Console.WriteLine($"Выходные данные после масштабирования и смещения: {output}");

                return output;
            }
            else
            {
                Console.WriteLine("Выполняется прямой проход в режиме предсказания.");

                // Используем скользящее среднее и дисперсию в режиме предсказания
                Tensor normalized = (input - movingMean) / (movingVariance + epsilon).Sqrt();
                Console.WriteLine($"Нормализованные данные (предсказание): {normalized}");

                // Масштабируем и смещаем нормализованные данные
                Tensor output = γ * normalized + β;
                Console.WriteLine($"Выходные данные после масштабирования и смещения (предсказание): {output}");

                return output;
            }
        }


        /// <summary>
        /// Устанавливает режим обучения или предсказания.
        /// </summary>
        /// <param name="training">Если true, слой работает в режиме обучения.</param>
        public void SetTrainingMode(bool training)
        {
            _training = training;
        }
    }
}
