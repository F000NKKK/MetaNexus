namespace MetaNexus.Lib.NeuralNetwork.Math.Functions
{
    /// <summary>
    /// Класс для активационных функций.
    /// </summary>
    public static class ActivationFunctions
    {
        /// <summary>
        /// Активационная функция ReLU.
        /// </summary>
        /// <param name="x">Входное значение.</param>
        /// <returns>Результат применения ReLU.</returns>
        public static double ReLU(double x)
        {
            return double.Max(0, x);
        }

        /// <summary>
        /// Производная активационной функции ReLU.
        /// </summary>
        public static double ReLUDerivative(double x)
        {
            return x > 0 ? 1 : 0;
        }

        /// <summary>
        /// Активационная функция Sigmoid.
        /// </summary>
        public static double Sigmoid(double x)
        {
            return 1 / (1 + double.Exp(-x));
        }

        /// <summary>
        /// Производная функции Sigmoid.
        /// </summary>
        public static double SigmoidDerivative(double x)
        {
            double sigmoid = Sigmoid(x);
            return sigmoid * (1 - sigmoid);
        }

        /// <summary>
        /// Активационная функция Tanh.
        /// </summary>
        public static double Tanh(double x)
        {
            return double.Tanh(x);
        }

        /// <summary>
        /// Производная функции Tanh.
        /// </summary>
        public static double TanhDerivative(double x)
        {
            return 1 - double.Pow(Tanh(x), 2);
        }
    }
}
