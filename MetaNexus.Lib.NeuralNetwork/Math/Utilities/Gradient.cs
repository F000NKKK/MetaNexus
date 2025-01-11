namespace MetaNexus.Lib.NeuralNetwork.Math.Utilities
{
    /// <summary>
    /// Класс для работы с градиентами в процессе обучения нейронной сети.
    /// </summary>
    public static class Gradient
    {
        /// <summary>
        /// Вычисляет градиент по методу конечных разностей.
        /// </summary>
        /// <param name="function">Функция, для которой вычисляется градиент.</param>
        /// <param name="x">Точка, в которой вычисляется градиент.</param>
        /// <param name="epsilon">Шаг для конечных разностей.</param>
        /// <returns>Градиент в виде массива значений.</returns>
        public static float[] Compute(Func<float[], float> function, float[] x, float epsilon = 1e-5f)
        {
            var gradient = new float[x.Length];

            for (int i = 0; i < x.Length; i++)
            {
                var xPlus = (float[])x.Clone();
                var xMinus = (float[])x.Clone();

                xPlus[i] += epsilon;
                xMinus[i] -= epsilon;

                gradient[i] = (function(xPlus) - function(xMinus)) / (2 * epsilon);
            }

            return gradient;
        }
    }
}