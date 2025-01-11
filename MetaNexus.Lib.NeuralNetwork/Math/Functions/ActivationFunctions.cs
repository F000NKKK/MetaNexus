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
        public static float ReLU(float x)
        {
            return float.Max(0f, x);
        }

        /// <summary>
        /// Производная активационной функции ReLU.
        /// </summary>
        public static float ReLUDerivative(float x)
        {
            return x > 0 ? 1 : 0;
        }

        /// <summary>
        /// Активационная функция Sigmoid.
        /// </summary>
        public static float Sigmoid(float x)
        {
            return 1 / (1 + float.Exp(-x));
        }

        /// <summary>
        /// Производная функции Sigmoid.
        /// </summary>
        public static float SigmoidDerivative(float x)
        {
            float sigmoid = Sigmoid(x);
            return sigmoid * (1 - sigmoid);
        }

        /// <summary>
        /// Активационная функция Tanh.
        /// </summary>
        public static float Tanh(float x)
        {
            return float.Tanh(x);
        }

        /// <summary>
        /// Производная функции Tanh.
        /// </summary>
        public static float TanhDerivative(float x)
        {
            return 1 - float.Pow(Tanh(x), 2);
        }
    }
}
