namespace MetaNexus.Lib.NeuralNetwork.ML.Layers.Abstractions
{
    /// <summary>
    /// Абстрактный класс для слоя нейронной сети.
    /// </summary>
    public abstract class Layer
    {
        public int Size { get; set; }

        /// <summary>
        /// Конструктор для слоя.
        /// </summary>
        /// <param name="size">Количество нейронов в слое.</param>
        public Layer(int size)
        {
            Size = size;
        }

        /// <summary>
        /// Метод для выполнения прямого прохода через слой.
        /// </summary>
        /// <param name="input">Входные данные для слоя.</param>
        /// <returns>Выходные данные после прохождения через слой.</returns>
        public abstract float[] Forward(float[] input);
    }
}
