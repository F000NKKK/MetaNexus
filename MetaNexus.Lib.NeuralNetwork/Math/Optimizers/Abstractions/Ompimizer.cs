﻿namespace MetaNexus.Lib.NeuralNetwork.Math.Optimizers.Abstractions
{
    /// <summary>
    /// Базовый класс для всех оптимизаторов.
    /// </summary>
    public abstract class Optimizer
    {
        /// <summary>
        /// Скорость обучения.
        /// </summary>
        protected double LearningRate { get; set; }

        /// <summary>
        /// Конструктор оптимизатора.
        /// </summary>
        /// <param name="learningRate">Начальная скорость обучения.</param>
        protected Optimizer(double learningRate)
        {
            LearningRate = learningRate;
        }

        /// <summary>
        /// Абстрактный метод для обновления параметров.
        /// </summary>
        /// <param name="parameters">Параметры модели.</param>
        /// <param name="gradients">Градиенты параметров.</param>
        public abstract void Update(double[] parameters, double[] gradients);
    }
}
