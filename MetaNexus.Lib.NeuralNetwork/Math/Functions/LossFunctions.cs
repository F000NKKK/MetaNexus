namespace MetaNexus.Lib.NeuralNetwork.Math.Functions
{
    /// <summary>
    /// Класс для вычисления функций потерь.
    /// </summary>
    public static class LossFunctions
    {
        /// <summary>
        /// Вычисляет среднеквадратичную ошибку (MSE).
        /// </summary>
        /// <param name="predictions">Предсказанные значения.</param>
        /// <param name="targets">Истинные значения.</param>
        /// <returns>MSE.</returns>
        public static float MeanSquaredError(float[] predictions, float[] targets)
        {
            return predictions.Zip(targets, (pred, target) => float.Pow(pred - target, 2)).Average();
        }

        /// <summary>
        /// Вычисляет среднюю абсолютную ошибку (MAE).
        /// </summary>
        /// <param name="predictions">Предсказанные значения.</param>
        /// <param name="targets">Истинные значения.</param>
        /// <returns>MAE.</returns>
        public static float MeanAbsoluteError(float[] predictions, float[] targets)
        {
            return predictions.Zip(targets, (pred, target) => float.Abs(pred - target)).Average();
        }

        /// <summary>
        /// Вычисляет кросс-энтропию.
        /// </summary>
        /// <param name="predictions">Предсказанные значения.</param>
        /// <param name="targets">Истинные значения.</param>
        /// <returns>Кросс-энтропия.</returns>
        public static float CrossEntropy(float[] predictions, float[] targets)
        {
            return -predictions.Zip(targets, (pred, target) => target * float.Log(pred + 1e-9f)).Sum();
        }
    }
}
