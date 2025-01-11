namespace MetaNexus.Lib.NeuralNetwork.ML.Models
{
    /// <summary>
    /// Класс, представляющий конфигурацию сети, полученную из JSON.
    /// </summary>
    public class NetworkConfig
    {
        public int InputSize { get; set; }  // Размер входных данных
        public List<LayerConfig> Layers { get; set; }
    }
}
