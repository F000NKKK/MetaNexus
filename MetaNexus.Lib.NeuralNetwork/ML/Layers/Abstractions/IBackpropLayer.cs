using MetaNexus.Lib.NeuralNetwork.Tensors;

namespace MetaNexus.Lib.NeuralNetwork.ML.Layers.Abstractions
{
    /// <summary>
    /// Интерфейс для слоев с поддержкой обратного распространения.
    /// </summary>
    public interface IBackpropLayer
    {
        /// <summary>
        /// Выполняет обратное распространение ошибки через слой.
        /// </summary>
        /// <param name="delta">Градиент ошибки от следующего слоя.</param>
        /// <param name="learningRate">Коэффициент обучения.</param>
        /// <returns>Градиент для предыдущего слоя.</returns>
        Tensor Backward(Tensor delta, float learningRate);
    }
}
