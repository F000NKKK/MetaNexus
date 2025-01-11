namespace MetaNexus.Lib.NeuralNetwork.Math.Utilities
{
    /// <summary>
    /// Класс для инициализации параметров нейронной сети.
    /// </summary>
    public static class Initializers
    {
        private static readonly Random Random = new Random(DateTime.Now.Millisecond);

        /// <summary>
        /// Инициализация весов методом Ксавье.
        /// </summary>
        /// <param name="size">Размер массива весов.</param>
        /// <param name="inputSize">Размер входного слоя.</param>
        /// <param name="outputSize">Размер выходного слоя.</param>
        /// <returns>Массив инициализированных весов.</returns>
        public static float[] Xavier(int size, int inputSize, int outputSize)
        {
            float stdDev = MathF.Sqrt(2.0f / (inputSize + outputSize)); // Используем MathF для работы с float
            return Enumerable.Range(0, size)
                .Select(_ => (float)Random.NextDouble() * stdDev * 2f - stdDev)
                .ToArray();
        }

        /// <summary>
        /// Инициализация весов методом Хе.
        /// </summary>
        /// <param name="size">Размер массива весов.</param>
        /// <param name="inputSize">Размер входного слоя.</param>
        /// <returns>Массив инициализированных весов.</returns>
        public static float[] He(int size, int inputSize)
        {
            float stdDev = MathF.Sqrt(2.0f / inputSize); // Используем MathF для работы с float
            return Enumerable.Range(0, size)
                .Select(_ => (float)Random.NextDouble() * stdDev * 2 - stdDev)
                .ToArray();
        }

        /// <summary>
        /// Инициализация весов случайными значениями из равномерного распределения.
        /// </summary>
        /// <param name="size">Размер массива весов.</param>
        /// <param name="min">Минимальное значение.</param>
        /// <param name="max">Максимальное значение.</param>
        /// <returns>Массив инициализированных весов.</returns>
        public static float[] RandomUniform(int size, float min, float max)
        {
            return Enumerable.Range(0, size)
                .Select(_ => min + (max - min) * (float)Random.NextDouble())
                .ToArray();
        }
    }
}
