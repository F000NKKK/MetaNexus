namespace MetaNexus.Lib.NeuralNetwork.Math.Functions
{
    /// <summary>
    /// Класс для реализации регуляризаций в нейронных сетях.
    /// </summary>
    public static class Regularization
    {
        /// <summary>
        /// Применяет L1 регуляризацию.
        /// </summary>
        /// <param name="weights">Веса модели.</param>
        /// <param name="lambda">Параметр регуляризации.</param>
        /// <returns>Сумма L1 регуляризации.</returns>
        public static double L1Regularization(double[] weights, double lambda)
        {
            return lambda * weights.Sum(double.Abs);
        }

        /// <summary>
        /// Применяет L2 регуляризацию.
        /// </summary>
        /// <param name="weights">Веса модели.</param>
        /// <param name="lambda">Параметр регуляризации.</param>
        /// <returns>Сумма L2 регуляризации.</returns>
        public static double L2Regularization(double[] weights, double lambda)
        {
            return lambda * weights.Sum(w => w * w);
        }
    }
}
