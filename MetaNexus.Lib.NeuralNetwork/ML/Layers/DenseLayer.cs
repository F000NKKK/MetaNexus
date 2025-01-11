using MetaNexus.Lib.NeuralNetwork.ML.Layers.Abstractions;

namespace MetaNexus.Lib.NeuralNetwork.ML.Layers
{
    /// <summary>
    /// Класс для полносвязного слоя нейронной сети.
    /// </summary>
    public class DenseLayer : Layer
    {
        public string Activation { get; set; }

        /// <summary>
        /// Конструктор для полносвязного слоя.
        /// </summary>
        /// <param name="size">Количество нейронов в слое.</param>
        /// <param name="activation">Функция активации для слоя.</param>
        public DenseLayer(int size, string activation) : base(size)
        {
            Activation = activation;
        }

        /// <summary>
        /// Метод для выполнения прямого прохода через полносвязный слой.
        /// В данный момент возвращаем входные данные без изменений.
        /// </summary>
        /// <param name="input">Входные данные для слоя.</param>
        /// <returns>Выходные данные после прохождения через слой.</returns>
        public override float[] Forward(float[] input)
        {
            // Для простоты, возвращаем входные данные как выходные.
            // Здесь можно добавить логику вычислений с весами и функцией активации.
            return input.Select(x => x).ToArray();
        }
    }
}
