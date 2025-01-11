namespace MetaNexus.Lib.NeuralNetwork.Math.Tensor
{
    /// <summary>
    /// Класс для генерации случайных чисел на основе различных распределений.
    /// </summary>
    public static class Distributions
    {
        private static readonly Random Random = new Random();

        /// <summary>
        /// Генерирует значения из равномерного распределения.
        /// </summary>
        /// <param name="min">Минимальное значение.</param>
        /// <param name="max">Максимальное значение.</param>
        /// <param name="size">Количество значений.</param>
        /// <returns>Массив случайных значений из равномерного распределения.</returns>
        public static float[] Uniform(float min, float max, int size)
        {
            if (min >= max)
                throw new ArgumentException("Минимальное значение должно быть меньше максимального.");

            return Enumerable.Range(0, size)
                .Select(_ => min + (max - min) * (float)Random.NextDouble())
                .ToArray();
        }

        /// <summary>
        /// Генерирует значения из нормального распределения.
        /// </summary>
        /// <param name="mean">Среднее значение (математическое ожидание).</param>
        /// <param name="stdDev">Стандартное отклонение.</param>
        /// <param name="size">Количество значений.</param>
        /// <returns>Массив случайных значений из нормального распределения.</returns>
        public static float[] Normal(float mean, float stdDev, int size)
        {
            return Enumerable.Range(0, size)
                .Select(_ =>
                {
                    // Используем метод Бокса-Мюллера для генерации нормальных случайных чисел
                    float u1 = (float)(1.0 - Random.NextDouble());
                    float u2 = (float)(1.0 - Random.NextDouble());
                    float randStdNormal = -2.0f * MathF.Log(u1) * MathF.Sin(2.0f * float.Pi * u2);
                    return mean + stdDev * randStdNormal;
                })
                .ToArray();
        }

        /// <summary>
        /// Генерирует значения из экспоненциального распределения.
        /// </summary>
        /// <param name="lambda">Параметр λ (интенсивность).</param>
        /// <param name="size">Количество значений.</param>
        /// <returns>Массив случайных значений из экспоненциального распределения.</returns>
        public static float[] Exponential(float lambda, int size)
        {
            if (lambda <= 0)
                throw new ArgumentException("Параметр λ должен быть положительным.");

            return Enumerable.Range(0, size)
                .Select(_ => -MathF.Log(1 - (float)Random.NextDouble()) / lambda)
                .ToArray();
        }

        /// <summary>
        /// Генерирует значения из биномиального распределения.
        /// </summary>
        /// <param name="n">Число испытаний.</param>
        /// <param name="p">Вероятность успеха в каждом испытании.</param>
        /// <param name="size">Количество значений.</param>
        /// <returns>Массив случайных значений из биномиального распределения.</returns>
        public static int[] Binomial(int n, float p, int size)
        {
            if (n <= 0)
                throw new ArgumentException("Число испытаний должно быть положительным.");
            if (p < 0 || p > 1)
                throw new ArgumentException("Вероятность должна быть в диапазоне от 0 до 1.");

            return Enumerable.Range(0, size)
                .Select(_ =>
                {
                    int successes = 0;
                    for (int i = 0; i < n; i++)
                    {
                        if ((float)Random.NextDouble() < p)
                            successes++;
                    }
                    return successes;
                })
                .ToArray();
        }
    }
}
