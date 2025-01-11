namespace MetaNexus.Lib.NeuralNetwork.Math.Functions
{
    /// <summary>
    /// Класс для реализации различных методов регуляризации.
    /// </summary>
    public static class Regularization
    {
        /// <summary>
        /// Применяет L1 регуляризацию, добавляя штраф за абсолютное значение весов.
        /// </summary>
        /// <param name="weights">Веса модели.</param>
        /// <param name="lambda">Коэффициент регуляризации.</param>
        /// <returns>Сумма L1 регуляризации.</returns>
        public static float L1Regularization(float[] weights, float lambda)
        {
            return lambda * weights.Sum(float.Abs);
        }

        /// <summary>
        /// Применяет L2 регуляризацию, добавляя штраф за квадрат весов.
        /// </summary>
        /// <param name="weights">Веса модели.</param>
        /// <param name="lambda">Коэффициент регуляризации.</param>
        /// <returns>Сумма L2 регуляризации.</returns>
        public static float L2Regularization(float[] weights, float lambda)
        {
            return lambda * weights.Sum(w => w * w);
        }

        /// <summary>
        /// ElasticNet регуляризация, которая сочетает L1 и L2 регуляризации.
        /// </summary>
        /// <param name="weights">Веса модели.</param>
        /// <param name="l1Ratio">Доля L1 регуляризации (0 - только L2, 1 - только L1).</param>
        /// <param name="lambda">Коэффициент регуляризации.</param>
        /// <returns>Сумма ElasticNet регуляризации.</returns>
        public static float ElasticNetRegularization(float[] weights, float lambda, float l1Ratio)
        {
            return lambda * (L1Regularization(weights, l1Ratio) + L2Regularization(weights, (1 - l1Ratio)));
        }

        /// <summary>
        /// Dropout регуляризация, которая случайно отключает нейроны во время обучения.
        /// </summary>
        /// <param name="inputs">Входные данные.</param>
        /// <param name="dropoutRate">Вероятность отключения нейрона (например, 0.5).</param>
        /// <returns>Модифицированный массив входных данных с отключенными элементами.</returns>
        public static float[] Dropout(float[] inputs, float dropoutRate, Random random)
        {
            return inputs.Select(input => random.NextDouble() < dropoutRate ? 0 : input).ToArray();
        }

        /// <summary>
        /// L0 регуляризация, минимизирующая количество ненулевых весов (используется редко из-за сложности оптимизации).
        /// </summary>
        /// <param name="weights">Веса модели.</param>
        /// <param name="lambda">Коэффициент регуляризации.</param>
        /// <returns>Сумма L0 регуляризации.</returns>
        public static float L0Regularization(float[] weights, float lambda)
        {
            return lambda * weights.Count(w => w != 0);
        }

        /// <summary>
        /// MaxNorm регуляризация, ограничивающая норму весов.
        /// </summary>
        /// <param name="weights">Веса модели.</param>
        /// <param name="maxNorm">Максимальная допустимая норма.</param>
        /// <returns>Модифицированные веса.</returns>
        public static float[] MaxNormRegularization(float[] weights, float maxNorm)
        {
            float norm = MathF.Sqrt(weights.Sum(w => w * w));
            if (norm > maxNorm)
            {
                float scale = maxNorm / norm;
                return weights.Select(w => w * scale).ToArray();
            }
            return weights;
        }

        /// <summary>
        /// Label Smoothing регуляризация, сглаживающая метки классов для уменьшения уверенности модели.
        /// </summary>
        /// <param name="labels">Массив меток классов (например, one-hot векторы).</param>
        /// <param name="smoothing">Уровень сглаживания (например, 0.1).</param>
        /// <returns>Сглаженные метки.</returns>
        public static float[] LabelSmoothing(float[] labels, float smoothing)
        {
            int numClasses = labels.Length;
            float smoothValue = smoothing / numClasses;
            return labels.Select(label => label * (1 - smoothing) + smoothValue).ToArray();
        }
    }
}
