using MetaNexus.Lib.NeuralNetwork.Tensors;

namespace MetaNexus.Lib.NeuralNetwork.ML.Layers.Abstractions
{
    /// <summary>
    /// Интерфейс для слоя нейронной сети. Он включает методы для получения и установки весов и смещений,
    /// а также для выполнения прямого прохода и инициализации весов и смещений.
    /// </summary>
    public interface ILayer : IBackpropLayer
    {
        /// <summary>
        /// Получить тензор весов.
        /// </summary>
        /// <returns>Тензор весов слоя.</returns>
        Tensor GetWeights();

        /// <summary>
        /// Установить новый тензор весов.
        /// </summary>
        /// <param name="newWeights">Новый тензор весов.</param>
        void SetWeights(Tensor newWeights);

        /// <summary>
        /// Получить тензор смещений.
        /// </summary>
        /// <returns>Тензор смещений слоя.</returns>
        Tensor GetBiases();

        /// <summary>
        /// Установить новый тензор смещений.
        /// </summary>
        /// <param name="newBiases">Новый тензор смещений.</param>
        void SetBiases(Tensor newBiases);

        /// <summary>
        /// Применить прямой проход через слой.
        /// </summary>
        /// <param name="input">Входной тензор (данные для обработки).</param>
        /// <returns>Результат прямого прохода (выходной тензор).</returns>
        Tensor Forward(Tensor input);

        /// <summary>
        /// Инициализация весов и смещений случайными значениями.
        /// </summary>
        void InitializeWeightsAndBiases();

        /// <summary>
        /// Применяет функцию активации к входному тензору.
        /// </summary>
        /// <param name="input">Входной тензор.</param>
        /// <returns>Результат применения функции активации.</returns>
        Tensor ApplyActivation(Tensor input);

        /// <summary>
        /// Применяет производную функции активации к входному тензору.
        /// </summary>
        /// <param name="input">Входной тензор.</param>
        /// <returns>Результат применения производной функции активации.</returns>
        Tensor ApplyActivationPrime(Tensor input);
    }
}
