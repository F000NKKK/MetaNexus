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
        /// <param name="input">Входные данные для сети.</param>
        /// <returns>Выходные данные после прохождения через все слои сети.</returns>
        float[] Predict(float[] input);
    }
}
