using MetaNexus.Lib.NeuralNetwork.ML.Layers.Abstractions;

namespace MetaNexus.Lib.NeuralNetwork.ML.Layers
{
    /// <summary>
    /// Класс для входного слоя нейронной сети.
    /// Входной слой просто передает входные данные без изменений.
    /// </summary>
    public class InputLayer : Layer
    {
        /// <summary>
        /// Конструктор для входного слоя.
        /// </summary>
        /// <param name="size">Количество нейронов во входном слое.</param>
        public InputLayer(int size) : base(size) { }

        /// <summary>
        /// Метод для выполнения прямого прохода через входной слой.
        /// </summary>
        /// <param name="input">Входные данные.</param>
        /// <returns>Те же самые входные данные.</returns>
        public override float[] Forward(float[] input)
        {
            return input; // Входной слой просто передает входные данные
        }
    }
}
