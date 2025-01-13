using MetaNexus.Lib.NeuralNetwork.ML.Layers.Abstractions;
using MetaNexus.Lib.NeuralNetwork.Tensors;

namespace MetaNexus.Lib.NeuralNetwork.ML.Abstractions
{
    /// <summary>
    /// Интерфейс для нейронной сети.
    /// </summary>
    public interface INetwork
    {
        /// <summary>
        /// Метод для выполнения прогноза (прямого прохода) через сеть.
        /// </summary>
        /// <param name="input">Входной тензор.</param>
        /// <returns>Выходной тензор после прохождения через все слои сети.</returns>
        Tensor Predict(Tensor input);

        /// <summary>
        /// Метод для обучения сети на основе входных данных и целевых значений.
        /// </summary>
        /// <param name="input">Входной тензор для обучения.</param>
        /// <param name="target">Целевой тензор для обучения.</param>
        /// <param name="learningRate">Коэффициент обучения для алгоритма обратного распространения.</param>
        void Train(Tensor input, Tensor target, float learningRate);

        /// <summary>
        /// Метод для добавления слоя в нейронную сеть.
        /// </summary>
        /// <param name="layer">Слой для добавления в сеть.</param>
        void AddLayer(ILayer layer);
    }
}
