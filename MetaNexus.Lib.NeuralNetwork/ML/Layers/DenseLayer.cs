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
        /// Здесь вычисляется результат работы с весами и смещениями.
        /// </summary>
        /// <param name="input">Входные данные для слоя.</param>
        /// <returns>Выходные данные после прохождения через слой.</returns>
        public override float[] Forward(float[] input)
        {
            // Пример простого вычисления выхода слоя с учетом весов и смещений
            float[] output = new float[Size];

            for (int i = 0; i < Size; i++)
            {
                output[i] = 0f;
                for (int j = 0; j < input.Length; j++)
                {
                    output[i] += input[j] * Weights[i];  // Суммирование произведений
                }
                output[i] += Biases[i];  // Добавление смещения
            }

            return output;
        }
    }
}
