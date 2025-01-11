namespace MetaNexus.Lib.NeuralNetwork.ML.Models
{
    /// <summary>
    /// Класс, представляющий конфигурацию одного слоя сети, полученную из JSON.
    /// </summary>
    public class LayerConfig
    {
        public string Type { get; set; }
        public int Size { get; set; }
        public string Activation { get; set; }
    }
}
