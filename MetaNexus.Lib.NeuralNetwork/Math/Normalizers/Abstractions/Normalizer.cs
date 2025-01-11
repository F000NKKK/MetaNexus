namespace MetaNexus.Lib.NeuralNetwork.Math.Normalizers.Abstractions
{
    /// <summary>
    /// Абстрактный класс для нормализаторов данных.
    /// </summary>
    public abstract class Normalizer
    {
        /// <summary>
        /// Нормализует переданные данные.
        /// </summary>
        /// <param name="data">Массив данных для нормализации.</param>
        /// <returns>Нормализованный массив данных.</returns>
        public abstract float[] Normalize(float[] data);

        /// <summary>
        /// Откатывает нормализацию, возвращая данные в их исходное состояние.
        /// </summary>
        /// <param name="normalizedData">Нормализованные данные.</param>
        /// <returns>Исходные данные.</returns>
        public abstract float[] Denormalize(float[] normalizedData);
    }
}
