namespace MetaNexus.Lib.NeuralNetwork.Math.Utilities
{
    /// <summary>
    /// Класс для инициализации параметров нейронной сети.
    /// </summary>
    public static class Initializers
    {
        private static readonly Random Random = new Random();

        /// <summary>
        /// Инициализация весов методом Ксавье.
        /// </summary>
        /// <param name="size">Размер массива весов.</param>
        /// <param name="inputSize">Размер входного слоя.</param>
        /// <param name="outputSize">Размер выходного слоя.</param>
        /// <returns>Массив инициализированных весов.</returns>
        public static double[] Xavier(int size, int inputSize, int outputSize)
        {
            double stdDev = double.Sqrt(2.0 / (inputSize + outputSize));
            return Enumerable.Range(0, size)
                .Select(_ => Random.NextDouble() * stdDev * 2 - stdDev)
                .ToArray();
        }

        /// <summary>
        /// Инициализация весов методом Хе.
        /// </summary>
        /// <param name="size">Размер массива весов.</param>
        /// <param name="inputSize">Размер входного слоя.</param>
        /// <returns>Массив инициализированных весов.</returns>
        public static double[] He(int size, int inputSize)
        {
            double stdDev = double.Sqrt(2.0 / inputSize);
            return Enumerable.Range(0, size)
                .Select(_ => Random.NextDouble() * stdDev * 2 - stdDev)
                .ToArray();
        }

        /// <summary>
        /// Инициализация весов случайными значениями из равномерного распределения.
        /// </summary>
        /// <param name="size">Размер массива весов.</param>
        /// <param name="min">Минимальное значение.</param>
        /// <param name="max">Максимальное значение.</param>
        /// <returns>Массив инициализированных весов.</returns>
        public static double[] RandomUniform(int size, double min, double max)
        {
            return Enumerable.Range(0, size)
                .Select(_ => min + (max - min) * Random.NextDouble())
                .ToArray();
        }
    }
}