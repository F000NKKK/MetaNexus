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
        public static double MeanSquaredError(double[] predictions, double[] targets)
        {
            return predictions.Zip(targets, (pred, target) => double.Pow(pred - target, 2)).Average();
        }

        /// <summary>
        /// Вычисляет среднюю абсолютную ошибку (MAE).
        /// </summary>
        /// <param name="predictions">Предсказанные значения.</param>
        /// <param name="targets">Истинные значения.</param>
        /// <returns>MAE.</returns>
        public static double MeanAbsoluteError(double[] predictions, double[] targets)
        {
            return predictions.Zip(targets, (pred, target) => double.Abs(pred - target)).Average();
        }

        /// <summary>
        /// Вычисляет кросс-энтропию.
        /// </summary>
        /// <param name="predictions">Предсказанные значения.</param>
        /// <param name="targets">Истинные значения.</param>
        /// <returns>Кросс-энтропия.</returns>
        public static double CrossEntropy(double[] predictions, double[] targets)
        {
            return -predictions.Zip(targets, (pred, target) => target * double.Log(pred + 1e-9)).Sum();
        }
    }
}
