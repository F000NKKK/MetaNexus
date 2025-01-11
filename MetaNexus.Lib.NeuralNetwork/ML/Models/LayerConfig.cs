namespace MetaNexus.Lib.NeuralNetwork.ML.Models
{
    /// <summary>
    /// Класс, представляющий конфигурацию одного слоя сети, полученную из JSON.
    /// </summary>
    public class LayerConfig
    {
        public string Type { get; set; }  // Тип слоя, например "input", "dense"
        public int Size { get; set; }  // Количество нейронов в слое
        public string Activation { get; set; }  // Функция активации для слоя
    }
}