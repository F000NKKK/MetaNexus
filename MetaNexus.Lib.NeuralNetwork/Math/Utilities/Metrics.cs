namespace MetaNexus.Lib.NeuralNetwork.Math.Utilities
{
    /// <summary>
    /// Класс для вычисления метрик качества модели.
    /// </summary>
    public static class Metrics
    {
        /// <summary>
        /// Вычисляет среднеквадратическую ошибку (MSE).
        /// </summary>
        /// <param name="predictions">Предсказанные значения.</param>
        /// <param name="targets">Истинные значения.</param>
        /// <returns>Среднеквадратическая ошибка.</returns>
        public static float MeanSquaredError(float[] predictions, float[] targets)
        {
            if (predictions.Length != targets.Length)
                throw new ArgumentException("Размеры предсказаний и истинных значений должны совпадать.");

            return predictions.Zip(targets, (p, t) => MathF.Pow(p - t, 2)).Average();
        }

        /// <summary>
        /// Вычисляет среднюю абсолютную ошибку (MAE).
        /// </summary>
        /// <param name="predictions">Предсказанные значения.</param>
        /// <param name="targets">Истинные значения.</param>
        /// <returns>Средняя абсолютная ошибка.</returns>
        public static float MeanAbsoluteError(float[] predictions, float[] targets)
        {
            if (predictions.Length != targets.Length)
                throw new ArgumentException("Размеры предсказаний и истинных значений должны совпадать.");

            return predictions.Zip(targets, (p, t) => MathF.Abs(p - t)).Average();
        }
    }
}